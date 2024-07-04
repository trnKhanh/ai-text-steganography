import os
from typing import Union

import torch
from transformers import LogitsProcessor

from seed_schemes import seed_scheme_factory
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
        self.seed_fn = seed_scheme_factory.get(
            seed_scheme,
            salt_key=salt_key,
            private_key=private_key,
        )
        self.window_length = window_length

        # Initialize RNG
        self.rng = torch.Generator(device=device)

        # Compute the ranges of each value in base
        self.ranges = torch.zeros((self.msg_base + 1), dtype=torch.int64)
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
        vocab_perm = torch.randperm(self.vocab_size, generator=self.rng)
        vocab_list = vocab_perm[self.ranges[value] : self.ranges[value + 1]]

        return vocab_list

    def _get_value(self, input_ids: torch.Tensor):
        """
        Check whether the token is in the valid list.
        """
        self._seed_rng(input_ids[:-1])
        vocab_perm = torch.randperm(self.vocab_size, generator=self.rng)

        cur_token = input_ids[-1]
        cur_id = (vocab_perm == cur_token).nonzero(as_tuple=True)[0]
        value = (cur_id < self.ranges).type(torch.int).argmax().item() - 1

        return value


class EncryptorLogitsProcessor(LogitsProcessor, BaseProcessor):
    def __init__(
        self,
        prompt_ids: torch.Tensor,
        msg: bytes,
        gamma: float,
        *args,
        **kwargs
    ):
        """
        Args:
            msg: message to hide in the text.
            gamma: bias add to scores of token in valid list.
        """
        super().__init__(*args, **kwargs)

        self.start_pos = []
        for i in range(prompt_ids.size(0)):
            self.start_pos.append(prompt_ids[i].size(0))
        self.raw_msg = msg
        self.msg = bytes_to_base(msg, self.msg_base)
        self.gamma = gamma

    def __call__(
        self, input_ids_batch: torch.LongTensor, scores_batch: torch.FloatTensor
    ):
        # If the whole message is hidden already, then just return the raw scores.

        for i, input_ids in enumerate(input_ids_batch):
            cur_pos = input_ids.size(0)
            msg_ptr = cur_pos - self.start_pos[0]
            if msg_ptr >= len(self.msg):
                continue
            scores_batch[i] = self._add_bias_to_valid_list(
                input_ids, scores_batch[i], self.msg[msg_ptr]
            )

        return scores_batch

    def _add_bias_to_valid_list(
        self, input_ids: torch.Tensor, scores: torch.Tensor, value: int
    ):
        """
        Add the bias (gamma) to the valid list tokens
        """
        ids = self._get_valid_list_ids(input_ids, value)
        scores[ids] = scores[ids] + self.gamma
        return scores

    def get_message_len(self):
        return len(self.msg)

    def validate(self, input_ids_batch: torch.Tensor):
        res = []
        for input_ids in input_ids_batch:
            values = []
            for i in range(self.start_pos[0], input_ids.size(0)):
                values.append(self._get_value(input_ids[: i + 1]))
            enc_msg = base_to_bytes(values, self.msg_base)
            cnt = 0
            for i in range(len(self.raw_msg)):
                if self.raw_msg[i] == enc_msg[i]:
                    cnt += 1
            res.append(cnt / len(self.raw_msg))
        

        return res


class DecryptorProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decrypt(self, input_ids_batch: torch.Tensor):
        """
        Decrypt the text sequences.
        """
        shift_msg = []
        for s in range(get_values_per_byte(self.msg_base)):
            msg = []
            bytes_msg = []
            for i, input_ids in enumerate(input_ids_batch):
                msg.append(list())
                for j in range(self.window_length + s, len(input_ids)):
                    # TODO: this could be slow. Considering reimplement this.
                    value = self._get_value(input_ids[: j + 1])
                    msg[i].append(value)

                bytes_msg.append(base_to_bytes(msg[i], self.msg_base))
            shift_msg.append(bytes_msg)

        return shift_msg
