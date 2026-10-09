"""Answer one audio question with SO-7B or SO-7B-MIX."""
import argparse
import logging

from data import validate_audio
from model import add_model_arguments, load_model, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--audio", required=True, help="16 kHz FOA WAV, or mono WAV with --mono.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mono", action="store_true", help="Use learned null spatial tokens (MIX only).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.mono and read_config(args.config)["variant"] != "mix":
        parser.error("--mono requires configs/so7b_mix.yaml.")
    validate_audio(args.audio, mono=args.mono)
    model = load_model(args)
    print(model.answer(args.audio, args.question, mono=args.mono))


if __name__ == "__main__":
    main()
