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


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
@pytest.mark.parametrize("variant,external_encoder", [("so", False), ("mix", False), ("so", True)])
def test_selective_download_preserves_adjacent_release_config(tmp_path, monkeypatch, backbone, variant, external_encoder):
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
    config = model.read_config(EXAMPLE / "configs" / (backbone + ("_mix.yaml" if variant == "mix" else ".yaml")))
    encoder_override = tmp_path / "shared-encoder.pt" if external_encoder else None
    if encoder_override:
        encoder_override.write_bytes(b"placeholder")
    checkpoint, encoder = model.resolve_checkpoint(config, encoder_checkpoint=encoder_override)
    assert len(calls) == 1
    request = calls[0]
    assert request["repo_id"] == config["repo_id"]
    assert request["revision"] == config["revision"]
    assert set(request["allow_patterns"]) == {
        model.CHECKPOINTS[backbone][variant],
        str(Path(model.CHECKPOINTS[backbone][variant]).parent / "train_args.json"),
        *([] if external_encoder else ["SO-Encoder/SO-Encoder.pt"]),
    }
    if external_encoder:
        assert encoder == encoder_override.resolve()
    assert Path(request["local_dir"]).is_relative_to(tmp_path / "hf-cache")
    assert not checkpoint.is_symlink()
    assert (checkpoint.resolve().parent / "train_args.json").is_file()
    assert encoder.is_file()
    # Reusing an explicit local release does not make another hub request.
    assert model.resolve_checkpoint(config, request["local_dir"], encoder_override) == (checkpoint, encoder)
    assert len(calls) == 1
    other_revision = dict(config, revision="another-revision")
    other_checkpoint, _ = model.resolve_checkpoint(other_revision, encoder_checkpoint=encoder_override)
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


@pytest.mark.parametrize("base", [None, "some-org/a-30b-model"])
def test_so30b_requires_local_base_before_source_or_download(tmp_path, monkeypatch, base):
    def unexpected(*args, **kwargs):
        pytest.fail("Source or checkpoint loading must follow local base validation")

    monkeypatch.setattr(model, "add_source", unexpected)
    monkeypatch.setattr(model, "resolve_checkpoint", unexpected)
    with pytest.raises(ValueError, match="downloaded base model directory"):
        model.SpatialOmniModel.from_config(EXAMPLE / "configs/so30b.yaml", base_model=base)


def test_so30b_rejects_source_without_its_entrypoint(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    source = source_checkout(tmp_path / "seven")
    monkeypatch.setattr(model, "add_source", lambda _: source)

    def unexpected(*args, **kwargs):
        pytest.fail("Checkpoint download must follow source branch validation")

    monkeypatch.setattr(model, "resolve_checkpoint", unexpected)
    with pytest.raises(ValueError, match="SO-30B source branch"):
        model.SpatialOmniModel.from_config(
            EXAMPLE / "configs/so30b.yaml", source_root=source, base_model=base,
        )


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
def test_loader_return_shape_and_mono_route(tmp_path, monkeypatch, backbone):
    source = source_checkout(tmp_path / "source")
    entrypoint = source / model.SO30B_ENTRYPOINT
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "adaptation.pt"
    encoder = tmp_path / "encoder.pt"
    events = []
    load_calls = []
    fake_model = types.SimpleNamespace(
        thinker=types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=2048),
            ),
        ),
    )
    processor = object()
    result = types.SimpleNamespace(missing_keys=[])
    release = types.ModuleType("spatial_omni.utils.release")
    release.load_release_settings = lambda *args, **kwargs: {"mixed_spatial_replay": True}
    transformers = types.ModuleType("transformers")
    transformers.set_seed = lambda seed: events.append(("seed", seed))
    trainer = types.ModuleType("train_so_qa")
    trainer.SpatialBeatsQACollator = lambda **kwargs: ("learned_null", kwargs)
    loader = types.ModuleType("scripts.batch_bench_so_qa")
    loader.unwrap_model = lambda instance: instance
    loader.SpatialBeatsEvalCollator = lambda **kwargs: ("eval", kwargs)

    def instantiate(runtime, path):
        events.append(("instantiate", path))
        load_calls.append(runtime)
        loaded = (fake_model, processor, {}, {}, result)
        return loaded + ({"loaded_keys": 884},) if backbone == "so30b" else loaded

    loader.instantiate_model_for_checkpoint = instantiate
    original_import = model.importlib.import_module

    def import_module(name):
        events.append(("import", name))
        if name == model.SO30B_ENTRYPOINT[:-3].replace("/", "."):
            return types.ModuleType(name)
        if name == "scripts.batch_bench_so_qa":
            return loader
        return original_import(name)

    monkeypatch.setitem(sys.modules, "spatial_omni.utils.release", release)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "train_so_qa", trainer)
    monkeypatch.setattr(model, "add_source", lambda _: source)
    monkeypatch.setattr(model, "resolve_checkpoint", lambda *args: (checkpoint, encoder))
    monkeypatch.setattr(model.importlib, "import_module", import_module)
    instance = model.SpatialOmniModel.from_config(
        EXAMPLE / "configs" / f"{backbone}_mix.yaml",
        source_root=source, base_model=base, device="cuda:1",
    )
    assert instance.model is fake_model
    assert instance.processor is processor
    assert instance.foa_collator == ("eval", {"processor": processor})
    assert load_calls[0].device == "cuda:1"
    assert load_calls[0].beats_checkpoint == str(encoder)
    if backbone == "so30b":
        assert load_calls[0].model_id == str(base.resolve())
        wrapper = ("import", model.SO30B_ENTRYPOINT[:-3].replace("/", "."))
        assert events.index(wrapper) < events.index(("import", "scripts.batch_bench_so_qa"))
        assert events.index(wrapper) < events.index(("instantiate", str(checkpoint)))
        assert instance.mono_collator == (
            "eval", {"processor": processor, "mono_audio_zero_spatial_tokens": True,
                     "zero_projected_spatial_dim": 2048},
        )
    else:
        assert instance.mono_collator == (
            "learned_null", {"processor": processor, "enable_mono_replay": True,
                             "include_generation_inputs": True},
        )
