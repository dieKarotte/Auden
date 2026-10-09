# Spatial-Omni

[Paper](https://arxiv.org/abs/2606.10738) · [Code](https://github.com/dieKarotte/Spatial-Omni) · [Checkpoints](https://huggingface.co/dieKarotte/Spatial-Omni) · [SO-Bench](https://huggingface.co/datasets/dieKarotte/SO-Bench)

Spatial-Omni answers questions about sounds and their spatial relationships. A
BEATs-based SO-Encoder reads four-channel first-order ambisonics (FOA), while the
base model retains its native audio encoder. Spatial features enter the language
model through a separate projector.

This example uses the published Spatial-Omni loader and collators for **SO-7B**
and **SO-7B-MIX**. MIX also accepts mono audio using learned null spatial tokens.
For training and other backbones, see the [Spatial-Omni repository](https://github.com/dieKarotte/Spatial-Omni#training).

## Installation

Use Linux, Python 3.11, and an NVIDIA GPU. The commands below run from an Auden
checkout. Install PyTorch for your CUDA version first; the tested environment
uses PyTorch 2.5.1 with CUDA 12.4, Transformers 4.52.0 and PEFT 0.17.1.

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e . -r examples/spatial_omni/requirements.txt

git clone https://github.com/dieKarotte/Spatial-Omni.git ../Spatial-Omni
git -C ../Spatial-Omni checkout 368295f7b5f585e8159c93acbd8628678f64013e
export SPATIAL_OMNI_ROOT="$(cd ../Spatial-Omni && pwd)"
cd examples/spatial_omni
```

Spatial-Omni is a separate source dependency; it is not installed as a Python
package. Keep the pinned checkout available when running this example. SDPA is
the default attention backend and does not require FlashAttention.

## Inference

Download a small FOA sample (about 1.6 MB):

```bash
hf download dieKarotte/SO-Bench \
  audio/test/foa_9975d396da0c90324197.wav \
  --repo-type dataset --revision 94af762afee08ecea2cf0769ecc2f4de86bc3484 \
  --local-dir data/SO-Bench

python infer.py --config configs/so7b.yaml \
  --audio data/SO-Bench/audio/test/foa_9975d396da0c90324197.wav \
  --question "Listen to the audio clip and answer based only on what you hear. Which sound source is located to the front-right and above? Answer in one sentence."
```

The first run downloads the selected adaptation checkpoint, its training
configuration, SO-Encoder and the base model. The YAML configs pin the adaptation
weights to HF revision `c466dad025629f6491a479b6c395dc4551649137`; both variants
require the base model specified in their released `train_args.json`.

| Config | Checkpoint | Input |
|---|---|---|
| `configs/so7b.yaml` | `SO-7B/so/SO-7B.pt` | FOA |
| `configs/so7b_mix.yaml` | `SO-7B/mix/SO-7B-MIX.pt` | FOA or explicit mono |

To use downloaded files, add `--checkpoint-dir /path/to/Spatial-Omni-weights`
and `--base-model /path/to/base-model`. Keep this layout:

```text
Spatial-Omni-weights/
  SO-Encoder/SO-Encoder.pt
  SO-7B/so/{SO-7B.pt,train_args.json}
  SO-7B/mix/{SO-7B-MIX.pt,train_args.json}
```

Only the selected variant is needed. Set `HF_ENDPOINT=https://hf-mirror.com`
before downloads if you use a mirror. Downloads are stored below
`$HF_HOME/auden/spatial_omni/<cache-key>` to keep weights beside their configuration
files. For offline use, pass the directory containing `SO-Encoder/` and `SO-7B/`
with `--checkpoint-dir` and a downloaded `--base-model`.

For MIX with FOA, change the config to `configs/so7b_mix.yaml`. For your own
single-channel recording:

```bash
python infer.py --config configs/so7b_mix.yaml --mono \
  --audio /path/to/mono.wav --question "What can you hear?"
```

Audio must be nonempty, **16 kHz**, and either four-channel FOA or, with
`--mono`, one channel. Preserve the released dataset's channel order, with W in
channel 0. The example does not resample, downmix or reorder channels. The model
uses at most the first 20 seconds. Spatial questions follow the DCASE coordinate
system: +x front, +y left, +z up; angles are in degrees.

## Batch prediction and scoring

`evaluate.py` reads JSONL records with `qa_id` (or `id`), `audio_path`, and
`question` or `prompt`. It uses `prompt` when supplied, including any answer
choices. A per-record `"mono": true` selects the MIX mono path. Audio paths are
relative to `--audio-root`, or to the manifest directory when it is omitted.

For three questions from SO-Bench, also download the QA split and two more clips:

```bash
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
```

Prediction runs default to 16 records and process one question at a time after
loading the model once. `--num-workers` controls parallel audio prechecks.
All QA metadata is preserved and a `prediction` field is added. Answers and
scoring metadata are not passed to the model. Output files must be new; if a run
fails, completed records remain in its output and the error reports the completed
count and output path. Use a new output path when retrying.

The scorer's dry run checks QA alignment and record count without an online judge.
For full evaluation, download the remaining SO-Bench audio and follow the
[official scoring protocol](https://github.com/dieKarotte/Spatial-Omni#evaluation).
The small example above is an integration check, not a full benchmark result.

## License and citation

Auden and Spatial-Omni code use Apache-2.0. The published Spatial-Omni weights use
[CC-BY-NC-SA-4.0](https://huggingface.co/dieKarotte/Spatial-Omni); the base model's
license also applies. SO-Bench recordings retain their underlying source terms.
The sample descriptor in `assets/samples.json` points to HF; audio and model
weights are not included here.

If you use Spatial-Omni, please cite:

```bibtex
@misc{zhu2026spatialomnispatialaudiounderstanding,
  title={Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding},
  author={Zhiyuan Zhu and Yixuan Chen and Yiwen Shao and Wenxiang Guo and Changhao Pan and Yu Zhang and Yuxiang Wang and Wei Liu and Houhua Zhang and Chengkuan Zeng and Wenbo Cheng and Yunxi Liu and Rui Yang and Steve Yves and Liefeng Bo and Zhou Zhao},
  year={2026},
  eprint={2606.10738},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2606.10738}
}
```
