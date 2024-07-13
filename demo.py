import torch
import gradio as gr

from utils import load_model
from stegno import generate, decrypt
from seed_scheme_factory import SeedSchemeFactory


def enc_fn(
    gen_model: str = "openai-community/gpt2",
    device: str = "cpu",
    prompt: str = "",
    msg: str = "",
    start_pos: int = 0,
    gamma: float = 2.0,
    msg_base: int = 2,
    seed_scheme: str = "dummy_hash",
    window_length: int = 1,
    private_key: int = 0,
    max_new_tokens_ratio: float = 2,
    num_beams: int = 4,
):
    model, tokenizer = load_model(gen_model, torch.device(device))
    text, msg_rate = generate(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        msg=str.encode(msg),
        start_pos_p=[start_pos],
        gamma=gamma,
        msg_base=msg_base,
        seed_scheme=seed_scheme,
        window_length=window_length,
        private_key=private_key,
        max_new_tokens_ratio=max_new_tokens_ratio,
        num_beams=num_beams,
    )
    return text, msg_rate


def dec_fn(
    gen_model: str = "openai-community/gpt2",
    device: str = "cpu",
    text: str = "",
    msg_base: int = 2,
    seed_scheme: str = "dummy_hash",
    window_length: int = 1,
    private_key: int = 0,
):
    model, tokenizer = load_model(gen_model, torch.device(device))
    msgs = decrypt(
        tokenizer=tokenizer,
        device=model.device,
        text=text,
        msg_base=msg_base,
        seed_scheme=seed_scheme,
        window_length=window_length,
        private_key=private_key,
    )
    msg_text = ""
    for i, msg in enumerate(msgs):
        msg_text += f"Shift {i}: {msg}\n\n"
    return msg_text


if __name__ == "__main__":
    enc = gr.Interface(
        fn=enc_fn,
        inputs=[
            gr.Textbox("openai-community/gpt2"),
            gr.Textbox("cpu"),
            gr.Textbox(),
            gr.Textbox(),
            gr.Number(),
            gr.Number(10.0),
            gr.Number(2),
            gr.Dropdown(value="dummy_hash", choices=SeedSchemeFactory.get_schemes_name()),
            gr.Number(1),
            gr.Number(),
            gr.Number(2),
            gr.Number(4),
        ],
        outputs=[
            gr.Textbox(label="Text containing message", show_label=True, show_copy_button=True),
            gr.Number(label="Percentage of message in text", show_label=True),
        ],
    )
    dec = gr.Interface(
        fn=dec_fn,
        inputs=[
            gr.Textbox("openai-community/gpt2"),
            gr.Textbox("cpu"),
            gr.Textbox(),
            gr.Number(2),
            gr.Dropdown(value="dummy_hash", choices=SeedSchemeFactory.get_schemes_name()),
            gr.Number(1),
            gr.Number(),
        ],
        outputs=[
            gr.Textbox(label="Message", show_label=True),
        ],
    )
    app = gr.TabbedInterface([enc, dec], ["Encrytion", "Decryption"])
    app.launch(share=True)
