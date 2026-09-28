# Third-Party Source and Weights

This repository does not include third-party source code. `python scripts/fetch_vendor.py` clones the projects below at
the commits pinned in `configs/vendor_sources.json` into `vendor/`, and checks every file this implementation uses
against `configs/vendor_sha256.json`. No pretrained weight is bundled.

| Directory after fetching | Upstream | Pinned commit |
| --- | --- | --- |
| vendor/pvic | https://github.com/fredzzhang/pvic | `35dd43a` |
| vendor/pvic/h_detr | https://github.com/fredzzhang/h-detr (PViC submodule) | `16042de` |
| vendor/pvic/detr | https://github.com/fredzzhang/detr (PViC submodule) | `7a152b4` |
| vendor/pvic/pocket | https://github.com/fredzzhang/pocket (PViC submodule) | `aeb1612` |
| vendor/pvic/vcoco | https://github.com/fredzzhang/vcoco (PViC submodule) | `cb13e3d` |
| vendor/pvic/hicodet | https://github.com/fredzzhang/hicodet (PViC submodule) | `047bbdc` |
| vendor/slhoi | https://github.com/MPI-Lab/SL-HOI | `64483cb` |

SL-HOI contains code from DINOv3 (https://github.com/facebookresearch/dinov3) and OpenAI CLIP
(https://github.com/openai/CLIP). `configs/vendor_patches/slhoi.patch` adds their license texts next to that code;
no other third-party file is modified. Each upstream license applies to its checked-out source.

The DINOv3 code and weights use Meta's DINOv3 agreement, not the SL-HOI MIT license. Publications using these
materials must acknowledge them. See the [official agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

Dataset images/annotations and pretrained weight access are separate from code licenses. Configure locally obtained,
authorized assets in `paths.local.sh`. Do not commit credentials or redistribute restricted assets based only on the
license of their download script.
