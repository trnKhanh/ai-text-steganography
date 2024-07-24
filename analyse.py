import os
from datetime import datetime
from copy import deepcopy
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


def load_msgs(msg_lens: list[int]):
    msgs = []
    c4_en = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    iterator = iter(c4_en)

    for length in tqdm(msg_lens, desc="Loading messages"):
        random_msg = torch.randint(256, (length,), generator=rng)
        msgs.append(["random", bytes(random_msg.tolist())])

        while True:
            readable_msg = next(iterator)["text"]
            try:
                msgs.append(["readable", readable_msg[:length].encode("ascii")])
                break
            except Exception as e:
                continue

    return msgs


def load_prompts(tokenizer, n: int, prompt_size: int):
    prompts = []

    c4_en = load_dataset("allenai/c4", "en", split="train", streaming=True)
    iterator = iter(c4_en)

    with tqdm(total=n, desc="Loading prompts") as pbar:
        while len(prompts) < n:
            text = next(iterator)["text"]
            input_ids = tokenizer.encode(text, return_tensors="pt")
            if input_ids.size(1) < prompt_size:
                continue
            truncated_text = tokenizer.batch_decode(input_ids[:, :prompt_size])[
                0
            ]
            prompts.append(truncated_text)
            pbar.update()

    return prompts


class AnalyseProcessor(object):
    params_names = [
        "msgs",
        "bases",
        "deltas",
    ]

    def __init__(
        self,
        save_file: str,
        save_freq: int | None = None,
        gen_model: str | None = None,
        judge_model: str | None = None,
        msgs: list[bytes] | None = None,
        bases: list[int] | None = None,
        deltas: list[float] | None = None,
        prompts: list[str] | None = None,
        repeat: int = 1,
        gen_params: dict | None = None,
        batch_size: int = 1,
    ):
        self.save_file = save_file
        self.save_freq = save_freq
        self.data = {
            "params": {
                "gen_model": gen_model,
                "judge_model": judge_model,
                "ptrs": {
                    "msgs": 0,
                    "bases": 0,
                    "deltas": 0,
                },
                "values": {
                    "msgs": msgs,
                    "bases": bases,
                    "deltas": deltas,
                },
                "prompts": prompts,
                "batch_size": batch_size,
                "repeat": repeat,
                "gen": gen_params,
            },
            "results": [],
        }
        self.__pbar = None
        self.last_saved = None
        self.skip_first = False

    def run(self, depth=0):
        if self.__pbar is None:
            total = 1
            for v in self.data["params"]["values"].keys():
                if v is None:
                    raise RuntimeError(f"values must not be None when running")

            initial = 0
            for param_name in self.params_names[::-1]:
                initial += total * self.data["params"]["ptrs"][param_name]
                total *= len(self.data["params"]["values"][param_name])

            if self.skip_first:
                initial += 1

            self.__pbar = tqdm(
                desc="Generating",
                total=total,
                initial=initial,
            )

        if depth < len(self.params_names):
            param_name = self.params_names[depth]

            while self.data["params"]["ptrs"][param_name] < len(
                self.data["params"]["values"][param_name]
            ):
                self.run(depth + 1)
                self.data["params"]["ptrs"][param_name] = (
                    self.data["params"]["ptrs"][param_name] + 1
                )

            self.data["params"]["ptrs"][param_name] = 0
            if depth == 0:
                self.save_data(self.save_file)
        else:
            if self.skip_first:
                self.skip_first = False
                return
            prompts = self.data["params"]["prompts"]

            msg_ptr = self.data["params"]["ptrs"]["msgs"]
            msg_type, msg = self.data["params"]["values"]["msgs"][msg_ptr]

            base_ptr = self.data["params"]["ptrs"]["bases"]
            base = self.data["params"]["values"]["bases"][base_ptr]

            delta_ptr = self.data["params"]["ptrs"]["deltas"]
            delta = self.data["params"]["values"]["deltas"][delta_ptr]

            model, tokenizer = ModelFactory.load_model(
                self.data["params"]["gen_model"]
            )
            l = 0
            while l < len(prompts):
                start = datetime.now()
                r = l + self.data["params"]["batch_size"]
                r = min(r, len(prompts))

                texts, msgs_rates, _ = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompts[l:r],
                    msg=msg,
                    msg_base=base,
                    delta=delta,
                    **self.data["params"]["gen"],
                )
                end = datetime.now()
                for i in range(len(texts)):
                    prompt_ptr = l + i
                    text = texts[i]
                    msg_rate = msgs_rates[i]
                    self.data["results"].append(
                        {
                            "ptrs": {
                                "prompts": prompt_ptr,
                                "msgs": msg_ptr,
                                "bases": base_ptr,
                                "deltas": delta_ptr,
                            },
                            "perplexity": ModelFactory.compute_perplexity(
                                self.data["params"]["judge_model"], text
                            ),
                            "text": text,
                            "msg_rate": msg_rate,
                            "run_time (ms)": (end - start).microseconds
                            / len(texts),
                        }
                    )
                l += self.data["params"]["batch_size"]

            postfix = {
                "base": base,
                "msg_len": len(msg),
                "delta": delta,
            }
            self.__pbar.refresh()
            if self.save_freq and (self.__pbar.n + 1) % self.save_freq == 0:
                self.save_data(self.save_file)

            if self.last_saved is not None:
                seconds = (datetime.now() - self.last_saved).seconds
                minutes = seconds // 60
                hours = minutes // 60
                minutes %= 60
                seconds %= 60
                postfix["last_saved"] = f"{hours}:{minutes}:{seconds} ago"

            self.__pbar.set_postfix(postfix)
            self.__pbar.update()

    def __get_mean(self, ptrs: dict, value_name: str):
        s = 0
        cnt = 0
        for r in self.data["results"]:
            msg_type, msg = self.data["params"]["values"]["msgs"][
                r["ptrs"]["msgs"]
            ]
            valid = True
            for k in ptrs:
                if (
                    (k in r["ptrs"] and r["ptrs"][k] != ptrs[k])
                    or (k == "msg_len" and len(msg) != ptrs[k])
                    or (k == "msg_type" and msg_type != ptrs[k])
                ):
                    valid = False
                    break

            if valid:
                s += r[value_name]
                cnt += 1
        if cnt == 0:
            cnt = 1
        return s / cnt

    def plot(self, figs_dir: str):
        os.makedirs(figs_dir, exist_ok=True)
        msg_set = set()
        for msg_type, msg in self.data["params"]["values"]["msgs"]:
            msg_set.add((msg_type, len(msg)))
        msg_set = sorted(msg_set)

        # Delta effect
        os.makedirs(os.path.join(figs_dir, "delta_effect"), exist_ok=True)
        for value_name in ["perplexity", "msg_rate"]:
            fig = plt.figure(dpi=300)
            for base_ptr, base in enumerate(
                self.data["params"]["values"]["bases"]
            ):
                for msg_type, msg_len in msg_set:
                    x = []
                    y = []
                    for delta_ptr, delta in enumerate(
                        self.data["params"]["values"]["deltas"]
                    ):
                        x.append(delta)
                        y.append(
                            self.__get_mean(
                                ptrs={
                                    "bases": base_ptr,
                                    "msg_type": msg_type,
                                    "msg_len": msg_len,
                                    "deltas": delta_ptr,
                                },
                                value_name=value_name,
                            )
                        )
                    plt.plot(
                        x,
                        y,
                        label=f"B={base}, msg_type={msg_type}, msg_len={msg_len}",
                    )
            plt.ylim(ymin=0)
            plt.legend()
            plt.savefig(
                os.path.join(figs_dir, "delta_effect", f"{value_name}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

        # Message length effect
        os.makedirs(os.path.join(figs_dir, "msg_len_effect"), exist_ok=True)
        for value_name in ["perplexity", "msg_rate"]:
            fig = plt.figure(dpi=300)
            for base_ptr, base in enumerate(
                self.data["params"]["values"]["bases"]
            ):
                for delta_ptr, delta in enumerate(
                    self.data["params"]["values"]["deltas"]
                ):
                    x = {}
                    y = {}
                    for msg_type, msg_len in msg_set:
                        if msg_type not in x:
                            x[msg_type] = []
                        if msg_type not in y:
                            y[msg_type] = []
                        x[msg_type].append(msg_len)
                        y[msg_type].append(
                            self.__get_mean(
                                ptrs={
                                    "bases": base_ptr,
                                    "msg_type": msg_type,
                                    "msg_len": msg_len,
                                    "deltas": delta_ptr,
                                },
                                value_name=value_name,
                            )
                        )
                    for msg_type in x:
                        plt.plot(
                            x[msg_type],
                            y[msg_type],
                            label=f"B={base}, msg_type={msg_type}, delta={delta}",
                        )
            plt.ylim(ymin=0)
            plt.legend()
            plt.savefig(
                os.path.join(figs_dir, "msg_len_effect", f"{value_name}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)
        print(f"Saved figures to {figs_dir}")

    def save_data(self, file_name: str):
        if file_name is None:
            return
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        data = deepcopy(self.data)
        for i in range(len(data["params"]["values"]["msgs"])):
            msg_type, msg = data["params"]["values"]["msgs"][i]
            if msg_type == "random":
                str_msg = base64.b64encode(msg).decode("ascii")
            else:
                str_msg = msg.decode("ascii")
            data["params"]["values"]["msgs"][i] = [msg_type, str_msg]

        with open(file_name, "w") as f:
            json.dump(data, f, indent=2)
        if self.__pbar is None:
            print(f"Saved AnalyseProcessor data to {file_name}")
        else:
            self.last_saved = datetime.now()

    def load_data(self, file_name: str):
        with open(file_name, "r") as f:
            self.data = json.load(f)
        for i in range(len(self.data["params"]["values"]["msgs"])):
            msg_type, str_msg = self.data["params"]["values"]["msgs"][i]
            if msg_type == "random":
                msg = base64.b64decode(str_msg)
            else:
                msg = str_msg.encode("ascii")
            self.data["params"]["values"]["msgs"][i] = [msg_type, msg]

        self.skip_first = len(self.data["results"]) > 0
        self.__pbar = None


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
        default=10,
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
        "--judge-model",
        type=str,
        default="gpt2",
        help="Model used to compute score perplexity of generated text",
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

    # Generate parameters
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to use sample or greedy search",
    )
    parser.add_argument(
        "--num-beams", type=int, default=1, help="How many beams to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for generating",
    )

    # Results
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat for each set of parameters, prompts and messages",
    )
    parser.add_argument(
        "--load-file",
        type=str,
        default=None,
        help="Where to load data for AnalyseProcessor",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=None,
        help="Where to save data for AnalyseProcessor",
    )
    parser.add_argument(
        "--save-freq", type=int, default=100, help="Save frequency"
    )
    parser.add_argument(
        "--figs-dir",
        type=str,
        default=None,
        help="Where to save figures",
    )

    return parser.parse_args()


def main(args):
    if not args.load_file:
        model, tokenizer = ModelFactory.load_model(args.gen_model)
        prompts = load_prompts(tokenizer, args.num_prompts, args.prompt_size)

        msgs_lens = []
        for i in np.linspace(
            args.msgs_lengths[0],
            args.msgs_lengths[1],
            int(args.msgs_lengths[2]),
            dtype=np.int64,
        ):
            for _ in range(args.msgs_per_length):
                msgs_lens.append(i)

        msgs = load_msgs(msgs_lens)

        processor = AnalyseProcessor(
            save_file=args.save_file,
            save_freq=args.save_freq,
            gen_model=args.gen_model,
            judge_model=args.judge_model,
            msgs=msgs,
            bases=args.bases,
            deltas=np.linspace(
                args.deltas[0], args.deltas[1], int(args.deltas[2])
            ).tolist(),
            prompts=prompts,
            batch_size=args.batch_size,
            gen_params=dict(
                start_pos_p=[0],
                seed_scheme="dummy_hash",
                window_length=1,
                min_new_tokens_ratio=1,
                max_new_tokens_ratio=1,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                repetition_penalty=1.0,
            ),
        )
        processor.save_data(args.save_file)
    else:
        processor = AnalyseProcessor(
            save_file=args.save_file,
            save_freq=args.save_freq,
        )
        processor.load_data(args.load_file)

    processor.run()
    processor.plot(args.figs_dir)

    # if args.figs_dir:
    #     process_results(results, args.figs_dir)


if __name__ == "__main__":
    args = create_args()
    main(args)
