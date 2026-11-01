from __future__ import annotations

from torch.utils.data import Dataset

import torch

from corisp import VCOCONullRoleIndex

_AGENT_ID_STRIDE = 1000

def _stable_agent_instance_ids(boxes: torch.Tensor) -> torch.Tensor:
    """Assign identities before augmentation can collapse distinct boxes."""

    identities = torch.empty(len(boxes), dtype=torch.long, device=boxes.device)
    representatives: list[torch.Tensor] = []
    for row, box in enumerate(boxes):
        identity = next(
            (
                index
                for index, representative in enumerate(representatives)
                if torch.allclose(box, representative, atol=1e-4, rtol=0.0)
            ),
            None,
        )
        if identity is None:
            identity = len(representatives)
            representatives.append(box.detach().clone())
        identities[row] = identity
    return identities

def include_official_vcoco_images(base_factory, image_ids) -> None:
    """Expand a converted V-COCO factory to an exact official image split."""

    if getattr(base_factory, "name", None) != "vcoco":
        raise ValueError("Official V-COCO expansion requires a V-COCO factory.")
    ordered_ids = [int(image_id) for image_id in image_ids]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("The official V-COCO split contains duplicate image ids.")
    dataset_ids = [int(image_id) for image_id in base_factory.dataset._image_ids]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("The converted V-COCO dataset contains duplicate image ids.")
    index_by_image_id = {
        image_id: index for index, image_id in enumerate(dataset_ids)
    }
    missing = [image_id for image_id in ordered_ids if image_id not in index_by_image_id]
    if missing:
        raise ValueError(
            f"Converted V-COCO data is missing {len(missing)} official images."
        )
    keep = [
        index_by_image_id[image_id] for image_id in ordered_ids
    ]
    annotations = getattr(base_factory.dataset, "_anno", None)
    if annotations is None:
        raise ValueError("Converted V-COCO data does not expose its annotations.")
    for index in keep:
        annotation = annotations[index]
        for field in ("boxes_h", "boxes_o"):
            boxes = annotation.get(field)
            if boxes is not None and len(boxes) == 0:
                # Pocket converts an empty Python list to shape (0,), while the
                # geometric transforms require every box tensor to be (N, 4).
                annotation[field] = torch.empty((0, 4), dtype=torch.float32)
    base_factory.dataset._keep = keep

class VCOCONullAwareDataFactory(Dataset):
    """Preserve PViC transforms while carrying null fillers through them.

    Null records temporarily use a negative class sentinel and duplicate the
    person box in the object slot.  Existing geometric transforms therefore
    apply exactly the same crop, flip, resize, and normalization.  Sentinels
    are split into dedicated target fields before the model sees the batch.
    """

    def __init__(
        self,
        base_factory,
        null_index: VCOCONullRoleIndex,
        *,
        include_null_only_images: bool = True,
    ) -> None:
        if getattr(base_factory, "name", None) != "vcoco":
            raise ValueError("Null-role augmentation is defined only for V-COCO.")
        self.dataset = base_factory.dataset
        self.transforms = base_factory.transforms
        self.name = base_factory.name
        self.null_index = null_index
        if include_null_only_images:
            null_image_ids = set(null_index.image_ids)
            extra = {
                index
                for index, image_id in enumerate(self.dataset._image_ids)
                if int(image_id) in null_image_ids
            }
            self.dataset._keep = sorted(set(self.dataset._keep).union(extra))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        # Pocket's generic tensor conversion represents an empty Python list as
        # float32.  Null-only V-COCO images therefore need an explicit class
        # dtype before sentinel injection and official class indexing.
        target["labels"] = target["actions"].long()
        target["object"] = target.pop("objects").long()
        image_id = int(self.dataset.image_id(index))
        records = self.null_index.for_image(image_id)
        if records:
            dtype = target["boxes_h"].dtype
            device = target["boxes_h"].device
            boxes = torch.tensor(
                [record.agent_box_xyxy for record in records],
                dtype=dtype,
                device=device,
            )
            sentinel = torch.tensor(
                [-(record.role_class_index + 1) for record in records],
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )
            target["boxes_h"] = torch.cat([target["boxes_h"], boxes], dim=0)
            target["boxes_o"] = torch.cat([target["boxes_o"], boxes], dim=0)
            target["labels"] = torch.cat([target["labels"], sentinel], dim=0)
            target["object"] = torch.cat(
                [
                    target["object"],
                    torch.zeros(
                        len(records),
                        dtype=target["object"].dtype,
                        device=target["object"].device,
                    ),
                ],
                dim=0,
            )

        # The inherited crop transform only keeps a fixed list of aligned
        # fields. Pack the source-agent identity into its integer object field
        # while augmentation filters the records, then restore both fields.
        # This prevents two different people clipped to the same crop boundary
        # from being merged into one semantic frame downstream.
        if target["object"].numel() and (
            bool((target["object"] < 0).any())
            or bool((target["object"] >= _AGENT_ID_STRIDE).any())
        ):
            raise ValueError("V-COCO object indices exceed the identity carrier.")
        agent_instance_ids = _stable_agent_instance_ids(target["boxes_h"])
        target["object"] = (
            target["object"].long()
            + agent_instance_ids * _AGENT_ID_STRIDE
        )
        image, target = self.transforms(image, target)
        packed_object = target["object"].long()
        agent_instance_ids = torch.div(
            packed_object, _AGENT_ID_STRIDE, rounding_mode="floor"
        )
        target["object"] = torch.remainder(
            packed_object, _AGENT_ID_STRIDE
        )
        null_mask = target["labels"] < 0
        target["null_agent_boxes"] = target["boxes_h"][null_mask]
        target["null_agent_instance_ids"] = agent_instance_ids[null_mask]
        target["null_role_classes"] = (
            -target["labels"][null_mask] - 1
        ).long()
        target["image_id"] = torch.tensor(image_id, dtype=torch.long)
        visible = ~null_mask
        for field in ("boxes_h", "boxes_o", "labels", "object"):
            target[field] = target[field][visible]
        target["agent_instance_ids"] = agent_instance_ids[visible]
        target["labels"] = target["labels"].long()
        target["object"] = target["object"].long()
        return image, target
