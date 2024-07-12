from typing import Union
import torch
import io

from cryptography.hazmat.primitives import hashes

from seed_scheme_factory import SeedSchemeFactory, SeedScheme


@SeedSchemeFactory.register("sha_left_hash")
class SHALeftHash(SeedScheme):
    def __init__(self, private_key: Union[int, None] = None, *args, **kwargs):
        self.private_key = (
            private_key.to_bytes(8, "big") if private_key is not None else None
        )

    def __call__(self, input_ids: torch.Tensor):
        buff = io.BytesIO()
        if self.private_key is not None:
            buff.write(self.private_key)
        for input_id in input_ids:
            buff.write(int(input_id.item()).to_bytes(8, "big"))
        buff.seek(0)
        input_ids_bytes = buff.read()

        digest = hashes.Hash(hashes.SHA224())
        digest.update(input_ids_bytes)
        hashed_value = digest.finalize()

        # Only take the first 8 bytes because seed in torch rng only accept int64 seed
        seed = int.from_bytes(hashed_value[:8], byteorder="big")

        return seed
