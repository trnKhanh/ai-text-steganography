from seed_scheme_factory import SeedSchemeFactory
import torch

if __name__ == "__main__":
    seed_fn = SeedSchemeFactory.get_instance("sha_left_hash", private_key=18)
    rng = torch.Generator()
    rng.manual_seed(1)
    input_ids = torch.randint(0, 2**32, (8,), generator=rng)
    print("input_ids =", input_ids)
    if seed_fn is not None:
        seed = seed_fn(input_ids)
        print("      ", 2**64-1)
        print("seed =", seed)

