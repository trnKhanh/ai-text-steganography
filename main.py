import os
import json
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
    # Generation Params
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams used in beam search",
    )
    parser.add_argument(
        "--max-new-tokens-ratio",
        type=float,
        default=2,
        help="Ratio of max new tokens to minimum tokens required to hide message",
    )
    # Input
    parser.add_argument(
        "--msg",
        type=str,
        default=None,
        help="Message or path to message to be hidden",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt or path to prompt used to generate text",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text or path to text containing the hidden message",
    )
    # Encryption params
    parser.add_argument(
        "--start-pos",
        type=int,
        nargs="+",
        default=[0],
        help="Start position to input the text (not including window length). If 2 integers are provided, choose the position randomly between the two values.",
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
            args.salt_key = int(f.readline())
        print(f"Read salt key from {args.salt_key}")
    else:
        args.salt_key = int(args.salt_key) if len(args.salt_key) > 0 else None

    if os.path.isfile(args.private_key):
        with open(args.private_key, "r") as f:
            args.private_key = int(f.readline())
        print(f"Read private key from {args.private_key}")
    else:
        args.private_key = (
            int(args.private_key) if len(args.private_key) > 0 else None
        )

    if args.encrypt:
        if len(args.prompt) == 0:
            raise ValueError("Prompt cannot be empty in encrypt mode")
        if len(args.msg) == 0:
            raise ValueError("Message cannot be empty in encrypt mode")

        if os.path.isfile(args.prompt):
            print(f"Read prompt from {args.prompt}")
            with open(args.prompt, "r") as f:
                args.prompt = "".join(f.readlines())

        if os.path.isfile(args.msg):
            print(f"Read message from {args.msg}")
            with open(args.msg, "rb") as f:
                args.msg = f.read()
        else:
            args.msg = bytes(args.msg)

        print("=" * os.get_terminal_size().columns)
        print("Encryption Parameters:")
        print(f"  GenModel: {args.gen_model}")
        print(f"  Prompt:")
        print("- " * (os.get_terminal_size().columns // 2))
        print(args.prompt)
        print("- " * (os.get_terminal_size().columns // 2))
        print(f"  Message:")
        print("- " * (os.get_terminal_size().columns // 2))
        print(args.msg)
        print("- " * (os.get_terminal_size().columns // 2))
        print(f"  Gamma: {args.gamma}")
        print(f"  Message Base: {args.msg_base}")
        print(f"  Seed Scheme: {args.seed_scheme}")
        print(f"  Window Length: {args.window_length}")
        print(f"  Salt Key: {args.salt_key}")
        print(f"  Private Key: {args.private_key}")
        print(f"  Max New Tokens Ratio: {args.max_new_tokens_ratio}")
        print(f"  Number of Beams: {args.num_beams}")
        print("=" * os.get_terminal_size().columns)
        text, msg_rate = generate(
            tokenizer=tokenizer,
            model=model,
            prompt=args.prompt,
            msg=args.msg,
            start_pos_p=args.start_pos,
            gamma=args.gamma,
            msg_base=args.msg_base,
            seed_scheme=args.seed_scheme,
            window_length=args.window_length,
            salt_key=args.salt_key,
            private_key=args.private_key,
            max_new_tokens_ratio=args.max_new_tokens_ratio,
            num_beams=args.num_beams,
        )
        print(f"Text contains message:")
        print("-" * (os.get_terminal_size().columns))
        print(text)
        print("-" * (os.get_terminal_size().columns))
        print(f"Successfully hide {msg_rate*100:.2f}% of the message")
        print("-" * (os.get_terminal_size().columns))

        if len(args.save_file) > 0:
            os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
            with open(args.save_file, "w") as f:
                f.write(text)
            print(f"Saved result to {args.save_file}")

    if args.decrypt:
        if len(args.text) == 0:
            raise ValueError("Text cannot be empty in decrypt mode")

        if os.path.isfile(args.text):
            print(f"Read text from {args.text}")
            with open(args.text, "r") as f:
                lines = f.readlines()
                args.text = "".join(lines)

        print("=" * os.get_terminal_size().columns)
        print("Decryption Parameters:")
        print(f"  GenModel: {args.gen_model}")
        print(f"  Message Base: {args.msg_base}")
        print(f"  Seed Scheme: {args.seed_scheme}")
        print(f"  Window Length: {args.window_length}")
        print(f"  Salt Key: {args.salt_key}")
        print(f"  Private Key: {args.private_key}")
        print(f"  Text:")
        print("- " * (os.get_terminal_size().columns // 2))
        print(args.text)
        print("- " * (os.get_terminal_size().columns // 2))
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
            print("-" * (os.get_terminal_size().columns))
            print(f"Shift {s}: ")
            print(msg[0])
        print("-" * (os.get_terminal_size().columns))


if __name__ == "__main__":
    args = create_args()
    main(args)
