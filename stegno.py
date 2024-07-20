from typing import Union

import torch
import transformers

from processors import EncryptorLogitsProcessor, DecryptorProcessor


def generate(
    tokenizer,
    model,
    prompt: str,
    msg: bytes,
    start_pos_p: list[int],
    delta: float,
    msg_base: int,
    seed_scheme: str,
    window_length: int = 1,
    salt_key: Union[int, None] = None,
    private_key: Union[int, None] = None,
    max_new_tokens_ratio: float = 2,
    num_beams: int = 4,
    repetition_penalty: float = 1.0,
):
    """
    Generate the sequence containing the hidden data.

    Args:
        tokenizer: tokenizer to use.
        model: generative model to use.
        prompt: input prompt.
        msg: message to hide in the text.
        delta: bias add to scores of token in valid list.
        msg_base: base of the message.
        seed_scheme: scheme used to compute the seed.
        window_length: length of window to compute the seed.
        salt_key: salt to add to the seed.
        private_key: private key used to compute the seed.

    """
    if len(start_pos_p) == 1:
        start_pos = start_pos_p[0]
    else:
        start_pos = torch.randint(
            start_pos_p[0], start_pos_p[1] + 1, (1,)
        ).item()
    start_pos = int(start_pos) + window_length

    tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    min_length = prompt_size + start_pos + logits_processor.get_message_len()
    max_length = prompt_size + int(
        start_pos + logits_processor.get_message_len() * max_new_tokens_ratio
    )
    max_length = min(max_length, tokenizer.model_max_length)
    min_length = min(min_length, max_length)
    output_tokens = model.generate(
        **tokenized_input,
        logits_processor=transformers.LogitsProcessorList([logits_processor]),
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        num_beams=num_beams,
        repetition_penalty=float(repetition_penalty),
    )

    output_tokens = output_tokens[:, prompt_size:]
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    output_tokens_post = tokenizer(output_text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    msg_rates, tokens_infos = logits_processor.validate(
        output_tokens_post.input_ids
    )

    return output_text, msg_rates[0], tokens_infos[0]


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
