"""CPU checks for source selection, release layout and lazy model loading."""

from pathlib import Path
import subprocess
import sys
import types

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

import model


def source_checkout(path):
    for name in ("scripts/batch_bench_so_qa.py", "spatial_omni/utils/release.py"):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")
    return path.resolve()


def clear_source_modules(monkeypatch):
    for name in ("spatial_omni", "scripts", "train_so_qa"):
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("variant", ["so", "mix"])
def test_selective_download_preserves_adjacent_release_config(tmp_path, monkeypatch, variant):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HOME = str(tmp_path / "hf-cache")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        for name in kwargs["allow_patterns"]:
            file = destination / name
            file.parent.mkdir(parents=True, exist_ok=True)
            # Tiny placeholder files test the layout, not model deserialization.
            file.write_text("{}", encoding="utf-8")
        return str(destination)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    config = model.read_config(EXAMPLE / "configs" / ("so7b_mix.yaml" if variant == "mix" else "so7b.yaml"))
    checkpoint, encoder = model.resolve_checkpoint(config)
    assert len(calls) == 1
    request = calls[0]
    assert request["repo_id"] == config["repo_id"]
    assert request["revision"] == config["revision"]
    assert set(request["allow_patterns"]) == {
        model.CHECKPOINTS[variant],
        str(Path(model.CHECKPOINTS[variant]).parent / "train_args.json"),
        "SO-Encoder/SO-Encoder.pt",
    }
    assert Path(request["local_dir"]).is_relative_to(tmp_path / "hf-cache")
    assert not checkpoint.is_symlink()
    assert (checkpoint.resolve().parent / "train_args.json").is_file()
    assert encoder.is_file()
    # Reusing an explicit local release does not make another hub request.
    assert model.resolve_checkpoint(config, request["local_dir"]) == (checkpoint, encoder)
    assert len(calls) == 1
    other_revision = dict(config, revision="another-revision")
    other_checkpoint, _ = model.resolve_checkpoint(other_revision)
    assert other_checkpoint != checkpoint


def test_selected_source_is_first_even_if_already_on_pythonpath(tmp_path, monkeypatch):
    selected = source_checkout(tmp_path / "selected")
    old = source_checkout(tmp_path / "old")
    clear_source_modules(monkeypatch)
    monkeypatch.setattr(sys, "path", [str(old), *sys.path, str(selected)])
    assert model.add_source(selected) == selected
    assert sys.path[0] == str(selected)
    assert sys.path.count(str(selected)) == 1


@pytest.mark.parametrize("name", ["spatial_omni", "scripts", "train_so_qa"])
def test_imported_module_from_other_checkout_is_rejected(tmp_path, monkeypatch, name):
    selected = source_checkout(tmp_path / "selected")
    old = source_checkout(tmp_path / "old")
    clear_source_modules(monkeypatch)
    foreign = types.ModuleType(name)
    if name == "scripts":
        # The released scripts directory is a namespace package.
        foreign.__path__ = [str(old / "scripts")]
    else:
        foreign.__file__ = str(old / (name + ".py"))
    monkeypatch.setitem(sys.modules, name, foreign)
    with pytest.raises(RuntimeError, match="already imported from another checkout"):
        model.add_source(selected)


@pytest.mark.parametrize("variant,saved_mix", [("so", True), ("mix", False)])
def test_mix_mismatch_fails_before_heavy_imports(tmp_path, monkeypatch, variant, saved_mix):
    config = model.read_config(EXAMPLE / "configs" / ("so7b_mix.yaml" if variant == "mix" else "so7b.yaml"))
    checkpoint = tmp_path / "checkpoint.pt"
    encoder = tmp_path / "encoder.pt"
    calls = []
    release = types.ModuleType("spatial_omni.utils.release")

    def load_release_settings(path, **kwargs):
        calls.append((path, kwargs))
        return {"mixed_spatial_replay": saved_mix}

    release.load_release_settings = load_release_settings
    monkeypatch.setitem(sys.modules, "spatial_omni.utils.release", release)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(model, "read_config", lambda _: config)
    monkeypatch.setattr(model, "add_source", lambda _: tmp_path)
    monkeypatch.setattr(model, "resolve_checkpoint", lambda *_: (checkpoint, encoder))
    with pytest.raises(ValueError, match="disagree on MIX mode"):
        model.SpatialOmniModel.from_config("unused.yaml", base_model="base-override")
    assert calls == [(checkpoint, {"model_id": "base-override", "encoder_checkpoint": encoder})]


@pytest.mark.parametrize("script", ["infer.py", "evaluate.py"])
def test_help_avoids_model_and_download_imports(script):
    code = """
import importlib.abc
from pathlib import Path
import runpy
import sys

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
            raise AssertionError("Unexpected heavyweight import: " + fullname)
        return None

target = Path(sys.argv[1])
sys.meta_path.insert(0, RejectHeavyImports())
sys.path.insert(0, str(target.parent))
sys.argv = [str(target), "--help"]
runpy.run_path(str(target), run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLE / script)],
        text=True, capture_output=True, check=True,
    )
    assert "--config" in result.stdout
    assert "--checkpoint-dir" in result.stdout
