# Spatial-Omni

[Paper](https://arxiv.org/abs/2606.10738) · [Code](https://github.com/dieKarotte/Spatial-Omni) · [Models](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni answers questions about sounds and their locations from four-channel
first-order ambisonic (FOA) audio. It adds an SO-Encoder and spatial projector
alongside the base model's native audio encoder.

This example supports SO-7B and SO-30B, including their MIX checkpoints. It uses
the model loaders from the corresponding Spatial-Omni branches.

## Setup

Use separate environments for the two backbones. Run the installation commands
from the Auden repository root.

| Model | Python | PyTorch / CUDA | Transformers |
|---|---|---|---|
| SO-7B | 3.11 | 2.5.1 / 12.4 | 4.52.0 |
| SO-30B | 3.12 | 2.9.0 / 12.8 | 5.0.0 |

SO-7B:

~~~bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e . -r examples/spatial_omni/requirements.txt
git clone --branch main https://github.com/dieKarotte/Spatial-Omni.git ../Spatial-Omni-7B
git -C ../Spatial-Omni-7B checkout 368295f7b5f585e8159c93acbd8628678f64013e
export SPATIAL_OMNI_ROOT="$(cd ../Spatial-Omni-7B && pwd)"
~~~

SO-30B:

~~~bash
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e . -r examples/spatial_omni/requirements-so30b.txt
git clone --branch SO-30B https://github.com/dieKarotte/Spatial-Omni.git ../Spatial-Omni-30B
git -C ../Spatial-Omni-30B checkout ba33a4c4866d6a0ed98fa0361644d36abc801fad
export SPATIAL_OMNI_ROOT="$(cd ../Spatial-Omni-30B && pwd)"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 --local-dir examples/spatial_omni/checkpoints/base-30b
export SO_BASE_MODEL="$PWD/examples/spatial_omni/checkpoints/base-30b"
~~~

The 30B loader requires the complete base model in a local directory. BF16 weights
alone take about 60 GB; use an 80 GB GPU. Both backbones use SDPA by default.

Keep the source checkout available and select its matching environment for each
run. Set `HF_ENDPOINT=https://hf-mirror.com` before downloads if needed.

## Inference

From `examples/spatial_omni/`, download a 1.6 MB FOA sample:

~~~bash
cd examples/spatial_omni
hf download dieKarotte/SO-Bench \
  audio/test/foa_9975d396da0c90324197.wav \
  --repo-type dataset --revision 94af762afee08ecea2cf0769ecc2f4de86bc3484 \
  --local-dir data/SO-Bench

export AUDIO=data/SO-Bench/audio/test/foa_9975d396da0c90324197.wav
export QUESTION="Listen to the audio clip and answer based only on what you hear. Which sound source is located to the front-right and above? Answer in one sentence."
~~~

In the SO-7B environment:

~~~bash
python infer.py --config configs/so7b.yaml --audio "$AUDIO" --question "$QUESTION"
~~~

In the SO-30B environment:

~~~bash
python infer.py --config configs/so30b.yaml --base-model "$SO_BASE_MODEL" \
  --audio "$AUDIO" --question "$QUESTION"
~~~

The selected adaptation weights, SO-Encoder and their configuration are downloaded
from Hugging Face. SO-7B also loads its base model from the identifier in the release
configuration. Downloads stay in `$HF_HOME/auden/spatial_omni/`.

| Config | Checkpoint | Input |
|---|---|---|
| `so7b.yaml` | `SO-7B/so/SO-7B.pt` | FOA |
| `so7b_mix.yaml` | `SO-7B/mix/SO-7B-MIX.pt` | FOA or mono |
| `so30b.yaml` | `SO-30B/so/SO-30B.pt` | FOA |
| `so30b_mix.yaml` | `SO-30B/mix/SO-30B-MIX.pt` | FOA or mono |

All config paths are under `configs/`. Add `--mono` for single-channel audio with
a MIX checkpoint. SO-7B uses learned null spatial tokens; SO-30B uses the released
evaluator's zero-spatial-token path.

Audio must be nonempty and 16 kHz, with four FOA channels or one mono channel.
Preserve the released dataset's FOA channel order, with W in channel 0. The model
uses the first 20 seconds. Coordinates follow DCASE: +x front, +y left, +z up;
angles are in degrees.

For local weights, pass `--checkpoint-dir` pointing to the HF release directory
containing `SO-7B/` or `SO-30B/` and `SO-Encoder/`. Each adaptation file needs its
adjacent `train_args.json`. Use `--encoder-checkpoint` if SO-Encoder is stored
elsewhere, and `--base-model` for a local base.

## Batch prediction

Download the QA records and the two remaining clips for a three-question check:

~~~bash
hf download dieKarotte/SO-Bench qa/test.jsonl \
  audio/test/foa_413d6010b124f7adcc0b.wav \
  audio/test/foa_be5d2301ea1330fce366.wav \
  --repo-type dataset --revision 94af762afee08ecea2cf0769ecc2f4de86bc3484 \
  --local-dir data/SO-Bench

python evaluate.py --manifest data/SO-Bench/qa/test.jsonl \
  --audio-root data/SO-Bench --limit 3 --dry-run
python evaluate.py --config configs/so7b.yaml \
  --manifest data/SO-Bench/qa/test.jsonl --audio-root data/SO-Bench \
  --limit 3 --output outputs/so7b/predictions.jsonl
python "$SPATIAL_OMNI_ROOT/scripts/score_sobench.py" \
  --predictions-jsonl outputs/so7b/predictions.jsonl \
  --qa-root data/SO-Bench/qa --expected-examples 3 --dry-run
~~~

For 30B, select its config and add `--base-model "$SO_BASE_MODEL"` to the prediction
command. Choose a separate output path for each run.

The JSONL manifest needs `qa_id` or `id`, `audio_path`, and `question` or `prompt`.
Audio paths are relative to `--audio-root`, or the manifest directory when omitted.
A supplied prompt takes precedence over question, so it can include answer choices.
Use `"mono": true` on mono records. Predictions retain the original QA fields.

The default limit is 16 records; `--limit 0` processes the full manifest.
`--num-workers` controls audio prechecks. Existing output files are refused, and a
failed run leaves completed predictions in place. The scorer's dry run checks
alignment without calling an online judge.

See Spatial-Omni for [training](https://github.com/dieKarotte/Spatial-Omni#training)
and the [full evaluation protocol](https://github.com/dieKarotte/Spatial-Omni#evaluation).

## License and citation

Code uses Apache-2.0. Spatial-Omni weights use
[CC-BY-NC-SA-4.0](https://huggingface.co/dieKarotte/Spatial-Omni); base models and
recordings retain their respective licenses. Audio and weights are downloaded
from their source repositories.

~~~bibtex
@misc{zhu2026spatialomnispatialaudiounderstanding,
  title={Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding},
  author={Zhiyuan Zhu and Yixuan Chen and Yiwen Shao and Wenxiang Guo and Changhao Pan and Yu Zhang and Yuxiang Wang and Wei Liu and Houhua Zhang and Chengkuan Zeng and Wenbo Cheng and Yunxi Liu and Rui Yang and Steve Yves and Liefeng Bo and Zhou Zhao},
  year={2026},
  eprint={2606.10738},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
~~~
