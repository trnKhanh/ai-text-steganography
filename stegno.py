from typing import Union

import torch
import transformers

from processors import EncryptorLogitsProcessor, DecryptorProcessor


def generate(
    tokenizer,
    model,
    prompt: str | list[str],
    msg: bytes,
    start_pos_p: list[int],
    delta: float,
    msg_base: int,
    seed_scheme: str,
    window_length: int = 1,
    salt_key: Union[int, None] = None,
    private_key: Union[int, None] = None,
    min_new_tokens_ratio: float = 1,
    max_new_tokens_ratio: float = 2,
    do_sample: bool = True,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
):
    """
    Generate the sequence containing the hidden data. This supports batch input/output.

    Args:
        tokenizer: tokenizer to use.
        model: generative model to use.
        prompt: input prompt.
        msg: message to hide in the text.
        start_pos_p: start position to hide message.
        delta: bias add to scores of token in valid list.
        msg_base: base of the message.
        seed_scheme: scheme used to compute the seed.
        window_length: length of window to compute the seed.
        salt_key: salt to add to the seed.
        private_key: private key used to compute the seed.
        min_new_tokens_ratio: ratio between min generated tokens and required token length.
        min_new_tokens_ratio: ratio between max generated tokens and required token length.
        do_sample: whether to do sampling or greedy generation.
        num_beams: number of beams used in beam search.
        repetition_penalty: penalty to avoid repetitiveness.
        generator: generation used to genereate. This is mainly used to produce deterministic results.
    Returns:
        generated texts, hidden message rates, tokens information
    """
    if len(start_pos_p) == 1:
        start_pos = start_pos_p[0]
    else:
        start_pos = torch.randint(
            start_pos_p[0], start_pos_p[1] + 1, (1,)
        ).item()
    start_pos = int(start_pos) + window_length
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenized_input = tokenizer(prompt, return_tensors="pt", padding=True).to(
        model.device
    )
    prompt_size = tokenized_input.input_ids.size(1)

    logits_processor = EncryptorLogitsProcessor(
        prompt_ids=tokenized_input.input_ids,
        msg=msg,
        start_pos=start_pos,
        delta=delta,
        msg_base=msg_base,
        vocab=list(tokenizer.get_vocab().values()),
        tokenizer=tokenizer,
        device=model.device,
        seed_scheme=seed_scheme,
        window_length=window_length,
        salt_key=salt_key,
        private_key=private_key,
    )
    min_length = (
        prompt_size
        + start_pos
        + logits_processor.get_message_len() * min_new_tokens_ratio
    )
    max_length = (
        prompt_size
        + start_pos
        + logits_processor.get_message_len() * max_new_tokens_ratio
    )
    max_length = min(max_length, tokenizer.model_max_length)
    min_length = min(min_length, max_length)
    output_tokens = model.generate(
        **tokenized_input,
        logits_processor=transformers.LogitsProcessorList([logits_processor]),
        min_length=min_length,
        max_length=max_length,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=float(repetition_penalty),
        pad_token_id=tokenizer.eos_token_id,
        generator=generator,
    )
    tokenizer.padding_side = "right"

    output_tokens = output_tokens[:, prompt_size:]
    output_texts = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )
    output_tokens_post = tokenizer(
        output_texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(model.device)

    msg_rates, tokens_infos = logits_processor.validate(
        output_tokens_post.input_ids
    )

    return output_texts, msg_rates, tokens_infos


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
        delta: bias added to scores of valid list.
        seed_scheme: scheme used to compute the seed.
        window_length: length of window to compute the seed.
        salt_key: salt to add to the seed.
        private_key: private key used to compute the seed.
    Returns:
        shifted versions of the message
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
