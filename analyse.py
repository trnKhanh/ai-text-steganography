import os
import json
import base64
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import torch
from datasets import load_dataset
from model_factory import ModelFactory
from stegno import generate

rng = torch.Generator(device="cpu")
rng.manual_seed(0)


def load_msgs(msg_lens: list[int], file: str | None = None):
    msgs = None
    if file is not None and os.path.isfile(file):
        with open(file, "r") as f:
            msgs = json.load(f)
        if "readable" not in msgs and "random" not in msgs:
            msgs = None
        else:
            return msgs

    msgs = {
        "readable": [],
        "random": [],
    }

    c4_en = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    iterator = iter(c4_en)

    for length in tqdm(msg_lens, desc="Loading messages"):
        random_msg = torch.randint(256, (length,), generator=rng)
        base64_msg = base64.b64encode(bytes(random_msg.tolist())).decode(
            "ascii"
        )
        msgs["random"].append(base64_msg)

        while True:
            readable_msg = next(iterator)["text"]
            try:
                readable_msg[:length].encode("ascii")
                break
            except Exception as e:
                continue
        msgs["readable"].append(readable_msg[:length])

    return msgs


def load_prompts(n: int, prompt_size: int, file: str | None = None):
    prompts = None
    if file is not None and os.path.isfile(file):
        with open(file, "r") as f:
            prompts = json.load(f)
        return prompts

    prompts = []

    c4_en = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(c4_en)

    with tqdm(total=n, desc="Loading prompts") as pbar:
        while len(prompts) < n:
            text = next(iterator)["text"]
            if len(text) < prompt_size:
                continue
            prompts.append(text)
            pbar.update()

    return prompts


def create_args():
    parser = ArgumentParser()

    # messages
    parser.add_argument(
        "--msgs-file", type=str, default=None, help="Where messages are stored"
    )
    parser.add_argument(
        "--msgs-lengths",
        nargs=3,
        type=int,
        help="Range of messages' lengths. This is parsed in form: <start> <end> <num>",
    )
    parser.add_argument(
        "--msgs-per-length",
        type=int,
        default=5,
        help="Number of messages per length",
    )
    # prompts
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Where prompts are stored",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts",
    )
    parser.add_argument(
        "--prompt-size",
        type=int,
        default=50,
        help="Size of prompts (in tokens)",
    )
    # Others
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite prompts and messages files",
    )

    # Hyperparameters
    parser.add_argument(
        "--gen-model",
        type=str,
        default="gpt2",
        help="Model used to generate",
    )
    parser.add_argument(
        "--deltas",
        nargs=3,
        type=float,
        help="Range of delta. This is parsed in form: <start> <end> <num>",
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        type=int,
        help="Bases used in base encoding",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt2",
        help="Model used to compute score perplexity of generated text",
    )
    # Results
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat for each set of parameters, prompts and messages",
    )
    parser.add_argument(
        "--results-load-file",
        type=str,
        default=None,
        help="Where to load results",
    )
    parser.add_argument(
        "--results-save-file",
        type=str,
        default=None,
        help="Where to save results",
    )
    parser.add_argument(
        "--results-save-freq", type=int, default=100, help="Save frequency"
    )
    parser.add_argument(
        "--figs-dir",
        type=str,
        default=None,
        help="Where to save figures",
    )

    return parser.parse_args()


def get_results(args, prompts, msgs):
    model, tokenizer = ModelFactory.load_model(args.gen_model)
    results = []
    total_gen = (
        len(prompts)
        * int(args.deltas[2])
        * len(args.bases)
        * args.repeat
        * sum([len(msgs[k]) for k in msgs])
    )

    with tqdm(total=total_gen, desc="Generating") as pbar:
        for k in msgs:
            msg_type = k
            for msg in msgs[k]:
                msg_bytes = (
                    msg.encode("ascii")
                    if k == "readable"
                    else base64.b64decode(msg)
                )
                for base in args.bases:
                    for delta in np.linspace(
                        args.deltas[0], args.deltas[1], int(args.deltas[2])
                    ):
                        for prompt in prompts:
                            for _ in range(args.repeat):
                                text, msg_rate, tokens_info = generate(
                                    tokenizer=tokenizer,
                                    model=model,
                                    prompt=prompt,
                                    msg=msg_bytes,
                                    start_pos_p=[0],
                                    delta=delta,
                                    msg_base=base,
                                    seed_scheme="sha_left_hash",
                                    window_length=1,
                                    private_key=0,
                                    min_new_tokens_ratio=1,
                                    max_new_tokens_ratio=2,
                                    num_beams=4,
                                    repetition_penalty=1.5,
                                    prompt_size=args.prompt_size,
                                )
                                results.append(
                                    {
                                        "msg_type": msg_type,
                                        "delta": delta.item(),
                                        "base": base,
                                        "perplexity": ModelFactory.compute_perplexity(
                                            args.judge_model, text
                                        ),
                                        "msg_rate": msg_rate,
                                        "msg_len": len(msg_bytes),
                                    }
                                )
                                pbar.set_postfix(
                                    {
                                        "perplexity": results[-1]["perplexity"],
                                        "msg_rate": results[-1]["msg_rate"],
                                        "msg_len": len(msg_bytes),
                                        "delta": delta.item(),
                                        "base": base,
                                    }
                                )
                                if (
                                    len(results) + 1
                                ) % args.results_save_freq == 0:
                                    if args.results_save_file:
                                        os.makedirs(
                                            os.path.dirname(
                                                args.results_save_file
                                            ),
                                            exist_ok=True,
                                        )
                                        with open(
                                            args.results_save_file, "w"
                                        ) as f:
                                            json.dump(results, f)
                                        print(
                                            f"Saved results to {args.results_save_file}"
                                        )

                                pbar.update()
    return results


def process_results(results, save_dir):
    data = {
        "perplexities": {
            "random": {},
            "readable": {},
        },
        "msg_rates": {
            "random": {},
            "readable": {},
        },
    }
    for r in results:
        msg_type = r["msg_type"]
        base = r["base"]
        delta = r["delta"]
        msg_rate = r["msg_rate"]
        msg_len = r["msg_len"]
        perplexity = r["perplexity"]

        if (base, delta, msg_len) not in data["msg_rates"][msg_type]:
            data["msg_rates"][msg_type][(base, delta, msg_len)] = []
        data["msg_rates"][msg_type][(base, delta, msg_len)].append(msg_rate)

        if (base, delta, msg_len) not in data["perplexities"][msg_type]:
            data["perplexities"][msg_type][(base, delta, msg_len)] = []
        data["perplexities"][msg_type][(base, delta, msg_len)].append(
            perplexity
        )

    bases = {
        "perplexities": {
            "random": [],
            "readable": [],
        },
        "msg_rates": {
            "random": [],
            "readable": [],
        },
    }
    deltas = {
        "perplexities": {
            "random": [],
            "readable": [],
        },
        "msg_rates": {
            "random": [],
            "readable": [],
        },
    }
    msgs_lens = {
        "perplexities": {
            "random": [],
            "readable": [],
        },
        "msg_rates": {
            "random": [],
            "readable": [],
        },
    }
    values = {
        "perplexities": {
            "random": [],
            "readable": [],
        },
        "msg_rates": {
            "random": [],
            "readable": [],
        },
    }
    base_set = set()
    delta_set = set()
    msgs_lens_set = set()
    for metric in data:
        for msg_type in data[metric]:
            for k in data[metric][msg_type]:
                s = sum(data[metric][msg_type][k])
                cnt = len(data[metric][msg_type][k])
                data[metric][msg_type][k] = s / cnt

                bases[metric][msg_type].append(k[0])
                deltas[metric][msg_type].append(k[1])
                msgs_lens[metric][msg_type].append(k[2])
                values[metric][msg_type].append(s / cnt)
                base_set.add(k[0])
                delta_set.add(k[1])
                msgs_lens_set.add(k[2])

    for metric in data:
        for msg_type in data[metric]:
            bases[metric][msg_type] = np.array(
                bases[metric][msg_type], dtype=np.int64
            )
            deltas[metric][msg_type] = np.array(
                deltas[metric][msg_type], dtype=np.int64
            )
            msgs_lens[metric][msg_type] = np.array(
                msgs_lens[metric][msg_type], dtype=np.int64
            )

            values[metric][msg_type] = np.array(
                values[metric][msg_type], dtype=np.float64
            )

    os.makedirs(save_dir, exist_ok=True)
    for metric in data:
        for msg_type in data[metric]:
            fig = plt.figure(dpi=300)
            s = lambda x: 3.0 + x * (30 if metric == "msg_rates" else 10)
            plt.scatter(
                bases[metric][msg_type],
                deltas[metric][msg_type],
                s(values[metric][msg_type]),
            )
            plt.savefig(
                os.path.join(save_dir, f"{metric}_{msg_type}_scatter.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    os.makedirs(os.path.join(save_dir, "delta_effect"), exist_ok=True)
    for metric in data:
        for msg_type in data[metric]:
            fig = plt.figure(dpi=300)
            for base_value in base_set:
                deltas_avg = np.array(list(sorted(delta_set)))
                values_avg = np.zeros_like(deltas_avg, dtype=np.float64)
                for i in range(len(deltas_avg)):
                    mask = (deltas[metric][msg_type] == deltas_avg[i]) & (
                        bases[metric][msg_type] == base_value
                    )
                    values_avg[i] = np.mean(values[metric][msg_type][mask])
                plt.plot(deltas_avg, values_avg, label=f"Base {base_value}")

            plt.legend()
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"delta_effect/{metric}_{msg_type}.pdf",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

    os.makedirs(os.path.join(save_dir, "msg_len_effect"), exist_ok=True)
    for metric in data:
        for msg_type in data[metric]:
            fig = plt.figure(dpi=300)
            for base_value in base_set:
                msgs_lens_avg = np.array(sorted(list(msgs_lens_set)))
                values_avg = np.zeros_like(msgs_lens_avg, dtype=np.float64)
                for i in range(len(msgs_lens_avg)):
                    mask = (msgs_lens[metric][msg_type] == msgs_lens_avg[i]) & (
                        bases[metric][msg_type] == base_value
                    )
                    values_avg[i] = np.mean(values[metric][msg_type][mask])

                plt.plot(msgs_lens_avg, values_avg, label=f"Base {base_value}")

            plt.legend()
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"msg_len_effect/{metric}_{msg_type}.pdf",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)


def main(args):
    if not args.results_load_file:
        prompts = load_prompts(
            args.num_prompts,
            args.prompt_size,
            args.prompts_file if not args.overwrite else None,
        )

        msgs_lens = []
        for i in np.linspace(
            args.msgs_lengths[0],
            args.msgs_lengths[1],
            int(args.msgs_lengths[2]),
            dtype=np.int64,
        ):
            for _ in range(args.msgs_per_length):
                msgs_lens.append(i)

        msgs = load_msgs(
            msgs_lens,
            args.msgs_file if not args.overwrite else None,
        )

        if args.msgs_file:
            if not os.path.isfile(args.msgs_file) or args.overwrite:
                os.makedirs(os.path.dirname(args.msgs_file), exist_ok=True)
                with open(args.msgs_file, "w") as f:
                    json.dump(msgs, f)
                print(f"Saved messages to {args.msgs_file}")
        if args.prompts_file:
            if not os.path.isfile(args.prompts_file) or args.overwrite:
                os.makedirs(os.path.dirname(args.prompts_file), exist_ok=True)
                with open(args.prompts_file, "w") as f:
                    json.dump(prompts, f)
                print(f"Saved prompts to {args.prompts_file}")
        results = get_results(args, prompts, msgs)
    else:
        with open(args.results_load_file, "r") as f:
            results = json.load(f)

    if args.results_save_file:
        os.makedirs(os.path.dirname(args.results_save_file), exist_ok=True)
        with open(args.results_save_file, "w") as f:
            json.dump(results, f)
        print(f"Saved results to {args.results_save_file}")

    if args.figs_dir:
        process_results(results, args.figs_dir)


if __name__ == "__main__":
    args = create_args()
    main(args)
