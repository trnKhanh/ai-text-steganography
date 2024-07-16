import torch

from seed_scheme_factory import SeedSchemeFactory, SeedScheme

@SeedSchemeFactory.register("dummy_hash")
class DummyHash(SeedScheme):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_ids: torch.Tensor):
        return int(input_ids[-1].item())

