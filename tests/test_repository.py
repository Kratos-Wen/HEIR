import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from examples.event_sets import run_example
from scripts.check_release import check_hygiene, check_links, check_manifests


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_example_uses_complete_set_matching():
    result = run_example()
    assert result["complete_set_map_percent"] == 100.
    assert result["incomplete_set_map_percent"] == 0.
    assert result["best_set"]["members"] == [
        {"entity_id": 1, "role": "target"}, {"entity_id": 2, "role": "instrument"}]
    assert 0 < result["best_set"]["score"] < 1
    assert 1 <= result["hypotheses"] <= 8


def test_example_command_runs_without_assets():
    result = subprocess.run([sys.executable, "-m", "examples.event_sets"], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["complete_set_map_percent"] == 100.


def test_local_documentation_links():
    check_links(ROOT)


def test_manifest_detects_edits_and_new_sources(tmp_path):
    (tmp_path / "configs").mkdir()
    script = tmp_path / "example.py"
    script.write_text("x = 1\n")
    check_manifests(tmp_path, update=True)
    check_manifests(tmp_path)
    script.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source mismatch"):
        check_manifests(tmp_path)
    check_manifests(tmp_path, update=True)
    (tmp_path / "extra.py").write_text("x = 3\n")
    with pytest.raises(ValueError, match="source mismatch"):
        check_manifests(tmp_path)


def test_hygiene_detects_private_paths_and_source_symlinks(tmp_path):
    script = tmp_path / "example.py"
    script.write_text("location = '" + "/ho" + "me/private/input'\n")
    with pytest.raises(ValueError, match="private absolute path"):
        check_hygiene(tmp_path)
    script.write_text("location = 'input'\n")
    (tmp_path / "linked.py").symlink_to(script)
    with pytest.raises(ValueError, match="Source symlink"):
        check_hygiene(tmp_path)


def test_fetched_vendor_is_outside_the_source_manifest_and_hygiene(tmp_path):
    (tmp_path / "configs").mkdir()
    vendor_data = tmp_path / "vendor" / "dependency" / "data"
    vendor_data.mkdir(parents=True)
    (vendor_data / "loader.py").write_text("path = '/" + "Users/upstream/example'\n")
    (tmp_path / "module.py").write_text("x = 1\n")
    assert check_hygiene(tmp_path) == 1
    check_manifests(tmp_path, update=True)
    source = json.loads((tmp_path / "configs/source_sha256.json").read_text())
    assert set(source) == {"module.py"}
    check_manifests(tmp_path)


def test_link_checker_detects_broken_and_external_local_paths(tmp_path):
    document = tmp_path / "README.md"
    document.write_text("[Missing](missing.md)\n")
    with pytest.raises(ValueError, match="broken local link"):
        check_links(tmp_path)
    document.write_text("[Parent](../)\n")
    with pytest.raises(ValueError, match="broken local link"):
        check_links(tmp_path)


def test_development_worktree_metadata_is_not_a_release_file(tmp_path):
    (tmp_path / ".git").write_text("gitdir: " + "/ho" + "me/project/.git/worktrees/code\n")
    (tmp_path / "example.py").write_text("x = 1\n")
    assert check_hygiene(tmp_path) == 1


def test_ci_is_unprivileged_and_pins_actions():
    workflow = yaml.load((ROOT / ".github/workflows/tests.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["test"]
    assert job["runs-on"] == "ubuntu-22.04"
    assert int(job["timeout-minutes"]) <= 20
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert job["env"][name] == "1"
    for step in job["steps"]:
        if "uses" in step:
            action, revision = step["uses"].rsplit("@", 1)
            assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
            if action == "actions/checkout":
                assert step["with"]["persist-credentials"] == "false"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "python -m pytest -q" in commands
    assert "python -m examples.event_sets" in commands
    assert "python scripts/check_release.py" in commands


def test_issue_templates_and_license_scope():
    for name in ("bug_report.yml", "question.yml"):
        template = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE" / name).read_text())
        assert template["name"] and template["description"]
        identifiers = [item["id"] for item in template["body"] if "id" in item]
        assert len(identifiers) == len(set(identifiers))
    assert "PolyForm Noncommercial License 1.0.0" in (ROOT / "LICENSE").read_text()
    assert "Third-party" in (ROOT / "NOTICE").read_text()
