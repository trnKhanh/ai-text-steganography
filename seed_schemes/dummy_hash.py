import torch

from seed_scheme_factory import SeedSchemeFactory, SeedScheme

@SeedSchemeFactory.register("dummy_hash")
class DummyHash(SeedScheme):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_ids: torch.Tensor):
        if input_ids.size(0) == 0:
            return  0
        return int(input_ids[-1].item())

