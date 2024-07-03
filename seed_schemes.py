from typing import Union, Callable

import torch


class SeedSchemeFactory:
    def __init__(self):
        self.seed_scheme_dict = dict()

    def register(self, name: str, seed_scheme: type):
        """
        Register the hash scheme by name. Hash scheme must be callable.

        Args:
            name: name of seed scheme.
            func: seed function.
        """
        self.seed_scheme_dict[name] = seed_scheme

    def get(self, name: str, **kwargs):
        """
        Get the hash scheme by name.

        Args:
            name: name of seed scheme.
        """
        return self.seed_scheme_dict[name](**kwargs)


class DummyHash:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_ids: torch.Tensor):
        return input_ids[-1].item()


seed_scheme_factory = SeedSchemeFactory()
seed_scheme_factory.register("dummy_hash", DummyHash)
