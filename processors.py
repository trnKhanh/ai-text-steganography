import os
from typing import Union

import torch
from transformers import LogitsProcessor

from seed_scheme_factory import SeedSchemeFactory
from utils import bytes_to_base, base_to_bytes, get_values_per_byte


class BaseProcessor(object):
    def __init__(
        self,
        msg_base: int,
        vocab: list[int],
        device: torch.device,
        seed_scheme: str,
        window_length: int = 1,
        salt_key: Union[int, None] = None,
        private_key: Union[int, None] = None,
    ):
        """
        Args:
            msg_base: base of the message.
            vocab: vocabulary list.
            device: device to load processor.
            seed_scheme: scheme used to compute the seed.
            window_length: length of window to compute the seed.
            salt_key: salt to add to the seed.
            private_key: private key used to compute the seed.
        """
        # Universal parameters
        self.msg_base = msg_base
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.device = device

        # Seed parameters
        seed_fn = SeedSchemeFactory.get_instance(
            seed_scheme,
            salt_key=salt_key,
            private_key=private_key,
        )
        if seed_fn is None:
            raise ValueError(f'Seed scheme "{seed_scheme}" is invalid')
        else:
            self.seed_fn = seed_fn

        self.window_length = window_length

        # Initialize RNG, always use cpu generator
        self.rng = torch.Generator(device="cpu")

        # Compute the ranges of each value in base
        self.ranges = torch.zeros((self.msg_base + 1), dtype=torch.int64).to(
            self.device
        )
        chunk_size = self.vocab_size / self.msg_base
        r = self.vocab_size % self.msg_base
        self.ranges[1:] = chunk_size
        self.ranges[1 : r + 1] += 1
        self.ranges = torch.cumsum(self.ranges, dim=0)

    def _seed_rng(self, input_ids: torch.Tensor):
        """
        Set the seed for the rng based on the current sequences.

        Args:
            input_ids: id in the input sequence.
        """
        seed = self.seed_fn(input_ids[-self.window_length :])
        self.rng.manual_seed(seed)

    def _get_valid_list_ids(self, input_ids: torch.Tensor, value: int):
        """
        Get ids of tokens in the valid list for the current sequences.
        """
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(
            self.vocab_size, generator=self.rng, device="cpu"
        ).to(self.device)
        vocab_list = vocab_perm[self.ranges[value] : self.ranges[value + 1]]

        return vocab_list

    def _get_value(self, input_ids: torch.Tensor):
        """
        Check whether the token is in the valid list.
        """
        self._seed_rng(input_ids[:-1])
        vocab_perm = torch.randperm(
            self.vocab_size, generator=self.rng, device="cpu"
        ).to(self.device)

        cur_token = input_ids[-1]
        cur_id = (vocab_perm == cur_token).nonzero(as_tuple=True)[0]
        value = (cur_id < self.ranges).type(torch.int).argmax().item() - 1

        return value


class EncryptorLogitsProcessor(LogitsProcessor, BaseProcessor):
    def __init__(
        self,
        prompt_ids: torch.Tensor,
        msg: bytes,
        delta: float,
        tokenizer,
        start_pos: int = 0,
        *args,
        **kwargs,
    ):
        """
        Args:
            msg: message to hide in the text.
            delta: bias add to scores of token in valid list.
        """
        super().__init__(*args, **kwargs)
        if prompt_ids.size(0) != 1:
            raise RuntimeError(
                "EncryptorLogitsProcessor does not support multiple prompts input."
            )

        self.prompt_size = prompt_ids.size(1)
        self.start_pos = start_pos

        self.raw_msg = msg
        self.msg = bytes_to_base(msg, self.msg_base)
        self.delta = delta
        self.tokenizer = tokenizer
        special_tokens = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
        ]
        special_tokens = [x for x in special_tokens if x is not None]
        self.special_tokens = torch.tensor(special_tokens, device=self.device)

    def __call__(
        self, input_ids_batch: torch.LongTensor, scores_batch: torch.FloatTensor
    ):
        # If the whole message is hidden already, then just return the raw scores.

        for i, input_ids in enumerate(input_ids_batch):
            cur_pos = input_ids.size(0)
            msg_ptr = cur_pos - (self.prompt_size + self.start_pos)
            if msg_ptr < 0 or msg_ptr >= len(self.msg):
                continue
            scores_batch[i] = self._add_bias_to_valid_list(
                input_ids, scores_batch[i], self.msg[msg_ptr]
            )

        return scores_batch

    def _add_bias_to_valid_list(
        self, input_ids: torch.Tensor, scores: torch.Tensor, value: int
    ):
        """
        Add the bias (delta) to the valid list tokens
        """
        ids = torch.cat(
            [self._get_valid_list_ids(input_ids, value), self.special_tokens]
        )

        scores[ids] = scores[ids] + self.delta
        return scores

    def get_message_len(self):
        return len(self.msg)

    def __map_input_ids(self, input_ids: torch.Tensor, base_arr, byte_arr):
        byte_enc_msg = [-1 for _ in range(input_ids.size(0))]
        base_enc_msg = [-1 for _ in range(input_ids.size(0))]
        base_msg = [-1 for _ in range(input_ids.size(0))]
        byte_msg = [-1 for _ in range(input_ids.size(0))]

        values_per_byte = get_values_per_byte(self.msg_base)
        start = self.start_pos % values_per_byte

        for i, b in enumerate(base_arr):
            base_enc_msg[i] = base_arr[i]
            byte_enc_msg[i] = byte_arr[(i - start) // values_per_byte]

        for i, b in enumerate(self.msg):
            if i + self.start_pos >= len(base_msg):
                break
            base_msg[i + self.start_pos] = b
            byte_msg[i + self.start_pos] = self.raw_msg[i // values_per_byte]

        return base_msg, byte_msg, base_enc_msg, byte_enc_msg

    def validate(self, input_ids_batch: torch.Tensor):
        res = []
        tokens_infos = []
        for input_ids in input_ids_batch:
            # Initialization
            base_arr = []

            # Loop and obtain values of all tokens
            for i in range(0, input_ids.size(0)):
                base_arr.append(self._get_value(input_ids[: i + 1]))

            values_per_byte = get_values_per_byte(self.msg_base)

            # Transform the values to bytes
            start = self.start_pos % values_per_byte
            byte_arr = base_to_bytes(base_arr[start:], self.msg_base)

            # Construct the
            cnt = 0
            enc_msg = byte_arr[self.start_pos // values_per_byte :]
            for i in range(min(len(enc_msg), len(self.raw_msg))):
                if self.raw_msg[i] == enc_msg[i]:
                    cnt += 1
            res.append(cnt / len(self.raw_msg))

            base_msg, byte_msg, base_enc_msg, byte_enc_msg = (
                self.__map_input_ids(input_ids, base_arr, byte_arr)
            )
            tokens = []
            input_strs = [self.tokenizer.decode([input]) for input in input_ids]
            for i in range(len(base_enc_msg)):
                tokens.append(
                    {
                        "token": input_strs[i],
                        "base_enc": base_enc_msg[i],
                        "byte_enc": byte_enc_msg[i],
                        "base_msg": base_msg[i],
                        "byte_msg": byte_msg[i],
                        "byte_id": (i - start) // values_per_byte,
                    }
                )
            tokens_infos.append(tokens)

        return res, tokens_infos


class DecryptorProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decrypt(self, input_ids_batch: torch.Tensor):
        """
        Decrypt the text sequences.
        """
        shift_msg = []
        for shift in range(get_values_per_byte(self.msg_base)):
            msg = []
            bytes_msg = []
            for i, input_ids in enumerate(input_ids_batch):
                msg.append(list())
                for j in range(shift, len(input_ids)):
                    # TODO: this could be slow. Considering reimplement this.
                    value = self._get_value(input_ids[: j + 1])
                    msg[i].append(value)

                bytes_msg.append(base_to_bytes(msg[i], self.msg_base))
            shift_msg.append(bytes_msg)

        return shift_msg
