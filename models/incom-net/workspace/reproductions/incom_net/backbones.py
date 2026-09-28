"""Frozen official DETR/CLIP feature adapters; no automatic random fallback."""

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import batched_nms

WORKSPACE = Path(__file__).resolve().parents[2]
PVIC = WORKSPACE / "reference_repos/pvic"
CLIP = WORKSPACE / "reference_repos/CLIP/clip"
CLIP_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def official_modules():
    # Namespaced CLIP import avoids the installed, third-party CLIP fork.
    name = "incom_openai_clip"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CLIP / "__init__.py", submodule_search_locations=[str(CLIP)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if "detr" not in sys.modules:
        sys.path.insert(0, str(PVIC))
    detr = importlib.import_module("detr")
    if Path(detr.__file__).resolve().parent != PVIC / "detr":
        raise ImportError("Unexpected DETR import; use an isolated reconstruction process")
    return importlib.import_module(name + ".model")


def select_proposals(result, image_hw, threshold=.05, min_instances=3, max_instances=15):
    """PViC/UPT selection policy, carrying ORIGINAL query IDs through NMS.

    Equivalent to PViC ops.prepare_region_proposals for positive-area boxes;
    unlike that helper, retains IDs to gather all three DETR decoder layers.
    """
    boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
    keep = batched_nms(boxes, scores, labels, .5)
    bx = boxes[keep].clone()
    bx[:, 0::2].clamp_(0, float(image_hw[1]))
    bx[:, 1::2].clamp_(0, float(image_hw[0]))
    positive = (bx[:, 2:] > bx[:, :2]).all(-1)
    keep, bx = keep[positive], bx[positive]
    sc, lb = scores[keep], labels[keep]
    selected = []
    for is_person in (True, False):
        candidates = torch.where((lb == 0) == is_person)[0]
        above = candidates[sc[candidates] >= threshold]
        if len(above) < min_instances:
            above = candidates[sc[candidates].argsort(descending=True)[:min_instances]]
        elif len(above) > max_instances:
            above = candidates[sc[candidates].argsort(descending=True)[:max_instances]]
        selected.append(above)
    positions = torch.cat(selected)
    return {"boxes": bx[positions], "scores": sc[positions], "labels": lb[positions],
            "query_ids": keep[positions]}


class FrozenCLIPLayers(nn.Module):
    def __init__(self, visual, levels=3):
        super().__init__()
        self.visual = visual.requires_grad_(False).eval()
        self.levels = levels
        if levels > len(visual.transformer.resblocks):
            raise ValueError("Insufficient visual layers")

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, normalized_clip_images):
        v = self.visual
        if normalized_clip_images.shape[-2:] != (v.input_resolution, v.input_resolution):
            raise ValueError("CLIP input resolution does not match the pretrained positional grid")
        x = v.conv1(normalized_clip_images.to(v.conv1.weight.dtype))
        grid = tuple(x.shape[-2:])
        x = x.flatten(2).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = v.ln_pre(torch.cat((cls, x), 1) + v.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        layers = []
        blocks = v.transformer.resblocks
        for index, block in enumerate(blocks):
            x = block(x)
            if index >= len(blocks) - self.levels:
                layers.append(x[1:].permute(1, 0, 2).float())
        return torch.stack(layers, dim=1), grid


class FrozenDETRLayers(nn.Module):
    def __init__(self, detector, postprocessor, levels=3):
        super().__init__()
        self.detector = detector.requires_grad_(False).eval()
        self.postprocessor = postprocessor
        self.levels = levels

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        from detr.util.misc import nested_tensor_from_tensor_list

        d = self.detector
        samples = nested_tensor_from_tensor_list(images)
        features, pos = d.backbone(samples)
        src, mask = features[-1].decompose()
        hs, _ = d.transformer(d.input_proj(src), mask, d.query_embed.weight, pos[-1])
        out = {"pred_logits": d.class_embed(hs[-1]), "pred_boxes": d.bbox_embed(hs[-1]).sigmoid()}
        sizes = torch.tensor([x.shape[-2:] for x in images], device=src.device)
        detections = self.postprocessor(out, sizes)
        records = []
        for i, (res, hw) in enumerate(zip(detections, sizes)):
            proposal = select_proposals(res, hw)
            ids = proposal["query_ids"]
            proposal["detector_layers"] = hs[-self.levels:, i, ids].float()
            proposal["cnn_tokens"] = src[i].flatten(1).T[~mask[i].flatten()].float()
            proposal["normalized_boxes"] = proposal["boxes"] / hw[[1, 0, 1, 0]]
            proposal["size"] = hw
            records.append(proposal)
        return records


class FrozenFeatureExtractor(nn.Module):
    def __init__(self, detector, clip):
        super().__init__()
        self.detector = detector
        self.clip = clip
        self.register_buffer("imagenet_mean", torch.tensor([.485, .456, .406])[:, None, None])
        self.register_buffer("imagenet_std", torch.tensor([.229, .224, .225])[:, None, None])
        self.register_buffer("clip_mean", torch.tensor([.48145466, .4578275, .40821073])[:, None, None])
        self.register_buffer("clip_std", torch.tensor([.26862954, .26130258, .27577711])[:, None, None])

    @torch.no_grad()
    def forward(self, imagenet_normalized_images):
        records = self.detector(imagenet_normalized_images)
        resolution = self.clip.visual.input_resolution
        # The same augmented view reaches both encoders; no additional center crop.
        images = []
        for image in imagenet_normalized_images:
            rgb = (image * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
            rgb = F.interpolate(rgb[None], (resolution, resolution), mode="bicubic", align_corners=False, antialias=True)[0]
            images.append((rgb - self.clip_mean) / self.clip_std)
        layers, grid = self.clip(torch.stack(images))
        for record, vlm in zip(records, layers):
            record["vlm_layers"] = vlm
            record["grid"] = grid
        return records


def build_detr_without_download():
    official_modules()
    from torchvision.models import resnet50
    from detr.models.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
    from detr.models.position_encoding import PositionEmbeddingSine
    from detr.models.transformer import Transformer
    from detr.models.detr import DETR, PostProcess

    # All weights are overwritten strictly; avoid an unrelated ResNet download.
    resnet = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = Joiner(BackboneBase(resnet, False, 2048, True), PositionEmbeddingSine(128, normalize=True))
    backbone.num_channels = 2048
    detector = DETR(backbone, Transformer(d_model=256, return_intermediate_dec=True), 80, 100, aux_loss=True)
    return detector, PostProcess()


def load_pretrained(detector_path, clip_path, device="cpu"):
    detector_path, clip_path = Path(detector_path), Path(clip_path)
    if not detector_path.is_file() or not clip_path.is_file():
        raise FileNotFoundError("Both actual pretrained files are required; random fallback is forbidden")
    if digest(clip_path) != CLIP_SHA256:
        raise ValueError("Expected original ViT-L/14@336px checkpoint, not ViT-L/14 at 224")
    clip_model = official_modules()
    detector, postprocessor = build_detr_without_download()
    checkpoint = torch.load(detector_path, map_location="cpu", weights_only=False)
    detector.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint

    jit = torch.jit.load(str(clip_path), map_location="cpu")
    visual_state = {k.removeprefix("visual."): v for k, v in jit.state_dict().items() if k.startswith("visual.")}
    visual = clip_model.VisionTransformer(336, 14, 1024, 24, 16, 768)
    visual.load_state_dict(visual_state, strict=True)
    del visual_state, jit
    return FrozenFeatureExtractor(FrozenDETRLayers(detector, postprocessor), FrozenCLIPLayers(visual)).to(device)
