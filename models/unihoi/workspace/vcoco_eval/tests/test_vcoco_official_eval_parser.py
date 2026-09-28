import importlib.util
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "vcoco_eval" / "scripts" / "evaluate_vcoco_official.py"
    spec = importlib.util.spec_from_file_location("_test_vcoco_official", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(scenario, offset):
    classes = (
        ("hit-obj", 10 + offset),
        ("hit-instr", 20 + offset),
        ("eat-obj", 30 + offset),
        ("eat-instr", 40 + offset),
        ("cut-obj", 50 + offset),
        ("cut-instr", 60 + offset),
    )
    lines = ["---------Reporting Role AP (%)------------------"]
    lines.extend(f"{name}: AP = {value:.2f} (#pos = 10)" for name, value in classes)
    lines.extend(
        (
            f"Average Role [{scenario}] AP = {35 + offset:.2f}",
            f'Average Role [{scenario}] AP = {36 + offset:.2f}, '
            'omitting the action "point"',
        )
    )
    return "\n".join(lines)


def test_parser_retains_overall_and_dual_role_metrics():
    module = _load_module()
    stdout = _block("scenario_1", 0) + "\n" + _block("scenario_2", 10)
    overall = module._parse_metrics(stdout)
    per_role = module._parse_role_metrics(stdout)
    dual = module._dual_role_summary(per_role)
    assert overall["role_ap_scenario_1_omit_point"] == 36.0
    assert overall["role_ap_scenario_2_omit_point"] == 46.0
    assert per_role["scenario_1"]["hit-obj"] == {"ap": 10.0, "positives": 10}
    assert dual["scenario_1"]["macro_ap"] == 35.0
    assert dual["scenario_2"]["macro_ap"] == 45.0


def test_parser_retains_official_agent_ap():
    module = _load_module()
    stdout = "\n".join(
        (
            "---------Reporting Agent AP (%)------------------",
            "hold: AP = 51.25 (#pos = 100)",
            "cut: AP = 63.50 (#pos = 50)",
            "Average Agent AP = 57.38",
            "---------------------------------------------",
        )
    )
    parsed = module._parse_agent_metrics(stdout)
    assert parsed["macro_ap"] == 57.38
    assert parsed["per_action"]["cut"] == {"ap": 63.5, "positives": 50}


def test_public_cache_template_is_deserialized_without_training_imports():
    module = _load_module()
    unpickler = module._VCOCOCacheUnpickler(io.BytesIO())
    cache_type = unpickler.find_class("utils", "CacheTemplate")
    record = cache_type(image_id=7)
    assert record["image_id"] == 7
    assert record["stand_agent"] == 0.0
    assert record["hold_obj"] == [0.0, 0.0, 0.1, 0.1, 0.0]
