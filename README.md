# CoRISP

**Compositional Role-aware Interaction Set Prediction**

CoRISP recovers the participants and functional roles that together form a person--action event. It combines role-conditioned interaction evidence with cardinality and role-multiplicity potentials in an exactly normalized set distribution. The implementation supports HEIR and V-COCO.

[Quick Start](#quick-start) | [Training and Evaluation](docs/REPRODUCTION.md) | [Method](docs/METHOD.md) | [Input Formats](docs/DATA_FORMATS.md) | [Contributing](CONTRIBUTING.md)

## What is included

This is a source-code release. Dataset annotations, images, semantic prototypes, support tables and pretrained or trained weights are not included. The synthetic example and numerical tests run without these assets or a GPU.

| Component | Entry point |
| --- | --- |
| Exact event normalization and set decoding | `evaluation/native.py` |
| HEIR training and prediction | `corisp_heir/train.py`, `corisp_heir/predict.py` |
| HEIR complete-set evaluation | `evaluation/heir_sets.py` |
| V-COCO training, caching and evaluation | `scripts/run_vcoco.sh` |
| Frozen visual encoders and detector integration | `integrations/` |

## Quick start

Run commands from the repository root. For the CPU example and test suite, use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m examples.event_sets
python -m pytest -q
python scripts/check_release.py
```

The example constructs a two-participant event, decodes synthetic potentials, and scores complete and incomplete predictions. It writes no files and downloads no assets. Its expected Set mAP is **100.0%** for the exact set and **0.0%** after a required participant is removed. These are interface checks, not benchmark results.

For training or image inference, follow [GPU installation](docs/INSTALLATION.md), provide the inputs in [Assets](docs/ASSETS.md), then use the [training and evaluation commands](docs/REPRODUCTION.md).

## Prediction and evaluation

HEIR set prediction retains up to eight count-state winners per person--action event and 100 sets per image, ranked by their normalized set probabilities. Role and HOI scores use relation marginals from the same model; HOI projection takes the maximum over roles. Select a checkpoint by validation Role mAP and use it consistently for test metrics and visualizations.

V-COCO uses native role slots and evaluates complete slot assignments under Scenarios 1 and 2. Official role AP and complete-set AP have separate evaluation entry points. [Method](docs/METHOD.md) and [Input Formats](docs/DATA_FORMATS.md) describe the model interfaces and score definitions.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/INSTALLATION.md) | CPU tests, CUDA setup and extension build |
| [Assets](docs/ASSETS.md) | Required local files and configuration |
| [Training and Evaluation](docs/REPRODUCTION.md) | HEIR and V-COCO commands |
| [Method](docs/METHOD.md) | Paper-to-code correspondence |
| [Input Formats](docs/DATA_FORMATS.md) | Entity identities, annotations and prediction records |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Dependency, asset and evaluation errors |
| [Third-party Source](THIRD_PARTY.md) | Dependency origins and licenses |

The core model is in `src/corisp/`; dataset integration uses the `corisp_heir` namespace. Run the source checkout from its root. The repository is not distributed as a standalone pip package.

## Repository branches

| Branch | Contents |
| --- | --- |
| `main` | CoRISP source, training and evaluation interfaces, documentation and tests |
| `reproductions/heir-baselines` | HEIR baseline evaluation tools and reproduction documentation |
| `reproductions/vcoco-baselines` | V-COCO baseline implementations, adapters and evaluation tools |

Each reproduction branch has its own README describing its scope, required assets and evaluation commands. Baseline reproductions are separate from the CoRISP implementation on `main`.

## License

Original CoRISP contributions are available under the [PolyForm Noncommercial License 1.0.0](LICENSE). This is a **noncommercial source-available release**. Third-party code retains its own license, including Meta's separate DINOv3 agreement. See [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for scope and attribution.
