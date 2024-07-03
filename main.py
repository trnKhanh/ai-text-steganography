import os
from argparse import ArgumentParser

import torch

from stegno import generate, decrypt
from utils import load_model


def create_args():
    parser = ArgumentParser()

    # Generative model
    parser.add_argument(
        "--gen-model",
        type=str,
        default="openai-community/gpt2",
        help="Generative model (LLM) used to generate text",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to load LLM"
    )
    # Stenography params
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Bias added to scores of tokens in valid list",
    )
    parser.add_argument(
        "--msg-base",
        type=int,
        default=2,
        help="Base of message",
    )
    parser.add_argument(
        "--seed-scheme",
        type=str,
        required=True,
        help="Scheme used to compute the seed",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=1,
        help="Length of window to compute the seed",
    )
    parser.add_argument(
        "--salt-key", type=str, default="", help="Path to salt key"
    )
    parser.add_argument(
        "--private-key", type=str, default="", help="Path to private key"
    )
    # Input
    parser.add_argument(
        "--msg", type=str, required=True, help="Path to file containing message"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Prompt used to generate text"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text contains the hidden message",
    )
    # Mode
    parser.add_argument(
        "--encrypt",
        action="store_true",
    )
    parser.add_argument(
        "--decrypt",
        action="store_true",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="",
        help="Where to save output",
    )

    return parser.parse_args()


def main(args):
    args.device = torch.device(args.device)
    model, tokenizer = load_model(args.gen_model, args.device)

    if os.path.isfile(args.salt_key):
        with open(args.salt_key, "r") as f:
            salt_key = int(f.readline())
    else:
        salt_key = None

    if os.path.isfile(args.private_key):
        with open(args.private_key, "r") as f:
            private_key = int(f.readline())
    else:
        private_key = None

    if args.encrypt:
        if len(args.prompt) == 0:
            raise ValueError("Prompt cannot be empty in encrypt mode")
        if os.path.isfile(args.msg):
            with open(args.msg, "rb") as f:
                msg = f.read()
        else:
            raise ValueError(f"Message file {args.msg} is not a file")

        print("=" * os.get_terminal_size().columns)
        print("Encryption Parameters:")
        print(f"  GenModel: {args.gen_model}")
        print(f"  Prompt: {args.prompt}")
        print(f"  Message: {msg}")
        print(f"  Gamma: {args.gamma}")
        print(f"  Message Base: {args.msg_base}")
        print(f"  Seed Scheme: {args.seed_scheme}")
        print(f"  Window Length: {args.window_length}")
        print(f"  Salt Key: {salt_key}")
        print(f"  Private Key: {private_key}")
        print("=" * os.get_terminal_size().columns)
        text = generate(
            tokenizer=tokenizer,
            model=model,
            prompt=args.prompt,
            msg=msg,
            gamma=args.gamma,
            msg_base=args.msg_base,
            seed_scheme=args.seed_scheme,
            window_length=args.window_length,
            salt_key=salt_key,
            private_key=private_key,
        )
        print(f"Text contains message:\n{text}")

        if os.path.isfile(args.save_file):
            with open(args.save_file, "w") as f:
                f.write(text)

        args.text = text

    if args.decrypt:
        if len(args.text) == 0:
            raise ValueError("Text cannot be empty in decrypt mode")
        if os.path.isfile(args.text):
            with open(args.text, "r") as f:
                lines = f.readlines()
                args.text = "".join(lines)
        print("=" * os.get_terminal_size().columns)
        print("Encryption Parameters:")
        print(f"  GenModel: {args.gen_model}")
        print(f"  Text: {args.text}")
        print(f"  Message Base: {args.msg_base}")
        print(f"  Seed Scheme: {args.seed_scheme}")
        print(f"  Window Length: {args.window_length}")
        print(f"  Salt Key: {salt_key}")
        print(f"  Private Key: {private_key}")
        print("=" * os.get_terminal_size().columns)
        msgs = decrypt(
            tokenizer=tokenizer,
            device=args.device,
            text=args.text,
            msg_base=args.msg_base,
            seed_scheme=args.seed_scheme,
            window_length=args.window_length,
            salt_key=args.salt_key,
            private_key=args.private_key,
        )
        print("Message:")
        for s, msg in enumerate(msgs):
            print(f"Shift {s}: {msg}")


if __name__ == "__main__":
    args = create_args()
    main(args)
