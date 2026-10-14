"""Load the published Spatial-Omni model and answer audio questions."""
import argparse
from contextlib import redirect_stdout
import importlib
import hashlib
import logging
import os
from pathlib import Path
import sys

from data import validate_audio

EXAMPLE_ROOT = Path(__file__).resolve().parent
SO30B_ENTRYPOINT = 'scripts/bench_test_generate_qwen3.py'
CHECKPOINTS = {
    "so7b": {"so": "SO-7B/so/SO-7B.pt", "mix": "SO-7B/mix/SO-7B-MIX.pt"},
    "so30b": {"so": "SO-30B/so/SO-30B.pt", "mix": "SO-30B/mix/SO-30B-MIX.pt"},
}


def add_source(source_root):
    source_root = source_root or os.environ.get("SPATIAL_OMNI_ROOT")
    if not source_root:
        raise ValueError("Set SPATIAL_OMNI_ROOT or --source-root to the Spatial-Omni checkout.")
    root = Path(source_root).expanduser().resolve()
    for filename in ("scripts/batch_bench_so_qa.py", "spatial_omni/utils/release.py"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Missing {filename} in Spatial-Omni checkout: {root}")
    for name in ("spatial_omni", "scripts", "train_so_qa"):
        module = sys.modules.get(name)
        if module is not None:
            paths = getattr(module, "__path__", ())
            origin = getattr(module, "__file__", None)
            locations = [Path(origin)] if origin else [Path(p) for p in paths]
            if any(not p.resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f"{name} was already imported from another checkout. Start a fresh Python process.")
    sys.path[:] = [str(root)] + [entry for entry in sys.path if entry != str(root)]
    return root


def read_config(config_path):
    import yaml

    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The example config must be a YAML mapping.")
    required = {"backbone", "variant", "repo_id", "revision", "device", "dtype", "attn_impl", "max_new_tokens", "seed"}
    if set(config) != required:
        raise ValueError(f"Config fields must be: {', '.join(sorted(required))}")
    if config["backbone"] not in CHECKPOINTS:
        raise ValueError("backbone must be so7b or so30b.")
    if config["variant"] not in ("so", "mix"):
        raise ValueError("variant must be so or mix.")
    if config["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype must be bfloat16, float16 or float32.")
    if config["attn_impl"] not in ("sdpa", "eager", "flash_attention_2"):
        raise ValueError("Unsupported attn_impl.")
    if type(config["max_new_tokens"]) is not int or config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    return config


def resolve_checkpoint(config, checkpoint_dir=None, encoder_checkpoint=None):
    filename = CHECKPOINTS[config["backbone"]][config["variant"]]
    if checkpoint_dir is None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HOME

        folder = str(Path(filename).parent)
        cache_key = hashlib.sha256(
            f'{config["repo_id"]}@{config["revision"]}'.encode()
        ).hexdigest()[:16]
        # The SO loader resolves symlinks before finding adjacent train_args.json.
        local_dir = Path(HF_HOME) / "auden" / "spatial_omni" / cache_key
        checkpoint_dir = snapshot_download(
            repo_id=config["repo_id"], revision=config["revision"],
            local_dir=local_dir,
            allow_patterns=[
                filename, f"{folder}/train_args.json",
                *([] if encoder_checkpoint else ["SO-Encoder/SO-Encoder.pt"]),
            ],
        )
    root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint = root / filename
    encoder = Path(encoder_checkpoint).expanduser().resolve() if encoder_checkpoint else root / "SO-Encoder/SO-Encoder.pt"
    for path in (checkpoint, checkpoint.parent / "train_args.json", encoder):
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {path}. Keep the HF repository directory layout.")
    return checkpoint, encoder


class SpatialOmniModel:
    """Inference adapter using Spatial-Omni's released loader and collators."""

    @classmethod
    def from_config(cls, config_path, source_root=None, checkpoint_dir=None,
                    base_model=None, device=None, encoder_checkpoint=None):
        config = read_config(config_path)
        if config["backbone"] == "so30b":
            if not base_model or not (Path(base_model).expanduser() / "config.json").is_file():
                raise ValueError("SO-30B needs --base-model pointing to a downloaded base model directory.")
            base_model = str(Path(base_model).expanduser().resolve())
        source = add_source(source_root)
        if config["backbone"] == "so30b" and not (source / SO30B_ENTRYPOINT).is_file():
            raise ValueError("SO-30B needs the Spatial-Omni SO-30B source branch.")
        checkpoint, encoder = resolve_checkpoint(config, checkpoint_dir, encoder_checkpoint)
        from spatial_omni.utils.release import load_release_settings

        settings = load_release_settings(
            checkpoint, model_id=base_model, encoder_checkpoint=encoder,
        )
        is_mix = bool(settings.get("mixed_spatial_replay", False))
        if is_mix != (config["variant"] == "mix"):
            raise ValueError("The selected config and checkpoint disagree on MIX mode.")
        from transformers import set_seed

        set_seed(config["seed"])
        if config["backbone"] == "so30b":
            importlib.import_module(SO30B_ENTRYPOINT[:-3].replace("/", "."))
        loader = importlib.import_module("scripts.batch_bench_so_qa")
        runtime = argparse.Namespace(
            model_id=base_model, beats_checkpoint=str(encoder),
            device=device or config["device"], device_map=None,
            dtype=config["dtype"], attn_impl=config["attn_impl"],
        )
        with redirect_stdout(sys.stderr):
            loaded = loader.instantiate_model_for_checkpoint(runtime, str(checkpoint))
        model, processor, _, state, result = loaded[:5]
        del state, loaded
        instance = cls()
        instance.model = model
        instance.processor = processor
        instance.config = config
        instance.is_mix = is_mix
        instance.loader = loader
        instance.foa_collator = loader.SpatialBeatsEvalCollator(processor=processor)
        instance.mono_collator = None
        if is_mix:
            if config["backbone"] == "so30b":
                thinker_config = loader.unwrap_model(model).thinker.config
                text_config = getattr(thinker_config, "text_config", None) or thinker_config
                instance.mono_collator = loader.SpatialBeatsEvalCollator(
                    processor=processor, mono_audio_zero_spatial_tokens=True,
                    zero_projected_spatial_dim=text_config.hidden_size,
                )
            else:
                from train_so_qa import SpatialBeatsQACollator

                instance.mono_collator = SpatialBeatsQACollator(
                    processor=processor, enable_mono_replay=True,
                    include_generation_inputs=True,
                )
        logging.info("Loaded %s; %d frozen base entries omitted.",
                     checkpoint, len(result.missing_keys))
        return instance

    def answer(self, audio_path, question, mono=False):
        """Generate text from 16 kHz FOA, or explicit single-channel MIX input."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string.")
        if mono and not self.is_mix:
            raise ValueError("Mono input requires a MIX config and checkpoint.")
        audio = Path(audio_path).expanduser().resolve()
        info = validate_audio(audio, mono=mono)
        logging.info("Audio: %s; %.2f s, %d Hz, %d channels",
                     audio, info.duration, info.samplerate, info.channels)
        if info.duration > 20:
            logging.warning("Spatial-Omni uses the first 20 seconds of the recording.")
        record = {"audio_path": str(audio), "question": question,
                  "prompt": question, "answer": ""}
        if mono:
            record["_replay_has_spatial"] = False
            batch = self.mono_collator([record])
        else:
            batch = self.foa_collator([record])
        inputs = self.loader.to_generation_inputs(
            batch, self.loader.get_model_device(self.model),
        )
        import torch

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, return_audio=False, do_sample=False, num_beams=1,
                max_new_tokens=self.config["max_new_tokens"],
            )
        prefix_length = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            generated[0, prefix_length:].cpu(), skip_special_tokens=True,
        ).strip()


def add_model_arguments(parser):
    parser.add_argument("--config", default=str(EXAMPLE_ROOT / "configs/so7b.yaml"))
    parser.add_argument("--source-root", help="Spatial-Omni source checkout; defaults to SPATIAL_OMNI_ROOT.")
    parser.add_argument("--checkpoint-dir", help="Local HF release root; otherwise download the selected variant.")
    parser.add_argument("--encoder-checkpoint", help="Use an SO-Encoder.pt outside the release directory.")
    parser.add_argument("--base-model", help="Local base directory (required for SO-30B), or a model ID for SO-7B.")
    parser.add_argument("--device", help="Override the config's device, e.g. cuda:0.")


def load_model(args):
    return SpatialOmniModel.from_config(
        args.config, source_root=args.source_root,
        checkpoint_dir=args.checkpoint_dir, base_model=args.base_model,
        device=args.device, encoder_checkpoint=getattr(args, "encoder_checkpoint", None),
    )
