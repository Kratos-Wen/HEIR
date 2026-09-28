"""Public instructions agree with the model's prediction entry point."""
from pathlib import Path

from evaluation.native import MAX_EVENT_SETS, MAX_IMAGE_SETS

ROOT = Path(__file__).resolve().parents[1]


def test_documented_set_budget():
    assert (MAX_EVENT_SETS, MAX_IMAGE_SETS) == (8, 100)
    reproduction = (ROOT / "docs/REPRODUCTION.md").read_text()
    assert "K=8 and a 100-set image budget" in reproduction


def test_prediction_example_uses_selected_checkpoint():
    reproduction = (ROOT / "docs/REPRODUCTION.md").read_text()
    assert 'export CORISP_CHECKPOINT=' in reproduction
    assert '--checkpoint "$CORISP_CHECKPOINT"' in reproduction
    assert "Select one checkpoint per trained model by validation Role mAP" in reproduction
    assert "epoch_030.pth" not in reproduction


def test_public_method_name_and_source_only_scope():
    readme = (ROOT / "README.md").read_text()
    assert readme.startswith("# CoRISP\n")
    assert "Compositional Role-aware Interaction Set Prediction" in readme
    assert "source-code release" in readme
    assert "weights are not included" in readme
