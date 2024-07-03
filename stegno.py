from typing import Union

import torch
import transformers

from processors import EncryptorLogitsProcessor, DecryptorProcessor


def generate(
    tokenizer,
    model,
    prompt: str,
    msg: bytes,
    gamma: float,
    msg_base: int,
    seed_scheme: str,
    window_length: int = 1,
    salt_key: Union[int, None] = None,
    private_key: Union[int, None] = None,
):
    """
    Generate the sequence containing the hidden data.

    Args:
        tokenizer: tokenizer to use.
        model: generative model to use.
        prompt: input prompt.
        msg: message to hide in the text.
        gamma: bias add to scores of token in valid list.
        msg_base: base of the message.
        seed_scheme: scheme used to compute the seed.
        window_length: length of window to compute the seed.
        salt_key: salt to add to the seed.
        private_key: private key used to compute the seed.

    """
    tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits_processor = EncryptorLogitsProcessor(
        prompt_ids=tokenized_input.input_ids,
        msg=msg,
        gamma=gamma,
        msg_base=msg_base,
        vocab=list(tokenizer.get_vocab().values()),
        device=model.device,
        seed_scheme=seed_scheme,
        window_length=window_length,
        salt_key=salt_key,
        private_key=private_key,
    )
    output_tokens = model.generate(
        **tokenized_input,
        logits_processor=transformers.LogitsProcessorList([logits_processor]),
        min_new_tokens=logits_processor.get_message_len(),
        max_new_tokens=logits_processor.get_message_len() * 2,
        do_sample=True,
        num_beams=4,
    )
    output_text = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )[0]

    return output_text


def decrypt(
    tokenizer,
    device: torch.device,
    text: str,
    msg_base: int,
    seed_scheme: str,
    window_length: int = 1,
    salt_key: Union[int, None] = None,
    private_key: Union[int, None] = None,
):
    """
    Extract the hidden data from the generated sequence.

    Args:
        tokenizer: tokenizer to use.
        text: text to decode.
        msg_base: base of the message.
        gamma: bias added to scores of valid list.
        seed_scheme: scheme used to compute the seed.
        window_length: length of window to compute the seed.
        salt_key: salt to add to the seed.
        private_key: private key used to compute the seed.
    """
    tokenized_input = tokenizer(text, return_tensors="pt").to(device)

    decryptor = DecryptorProcessor(
        msg_base=msg_base,
        vocab=list(tokenizer.get_vocab().values()),
        device=device,
        seed_scheme=seed_scheme,
        window_length=window_length,
        salt_key=salt_key,
        private_key=private_key,
    )

    msg = decryptor.decrypt(tokenized_input.input_ids)

    return msg

