import torch
import gradio as gr

from model_factory import ModelFactory
from stegno import generate, decrypt
from seed_scheme_factory import SeedSchemeFactory
from global_config import GlobalConfig


def enc_fn(
    gen_model: str,
    prompt: str,
    msg: str,
    start_pos: int,
    delta: float,
    msg_base: int,
    seed_scheme: str,
    window_length: int,
    private_key: int,
    do_sample: bool,
    min_new_tokens_ratio: float,
    max_new_tokens_ratio: float,
    num_beams: int,
    repetition_penalty: float,
):
    model, tokenizer = ModelFactory.load_model(gen_model)
    texts, msgs_rates, tokens_infos = generate(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        msg=str.encode(msg),
        start_pos_p=[start_pos],
        delta=delta,
        msg_base=msg_base,
        seed_scheme=seed_scheme,
        window_length=window_length,
        private_key=private_key,
        do_sample=do_sample,
        min_new_tokens_ratio=min_new_tokens_ratio,
        max_new_tokens_ratio=max_new_tokens_ratio,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    highlight_base = []
    for token in tokens_infos[0]:
        stat = None
        if token["base_msg"] != -1:
            if token["base_msg"] == token["base_enc"]:
                stat = "correct"
            else:
                stat = "wrong"
        highlight_base.append((repr(token["token"])[1:-1], stat))

    highlight_byte = []
    for i, token in enumerate(tokens_infos[0]):
        if i == 0 or tokens_infos[0][i - 1]["byte_id"] != token["byte_id"]:
            stat = None
            if token["byte_msg"] != -1:
                if token["byte_msg"] == token["byte_enc"]:
                    stat = "correct"
                else:
                    stat = "wrong"
            highlight_byte.append([repr(token["token"])[1:-1], stat])
        else:
            highlight_byte[-1][0] += repr(token["token"])[1:-1]

    return (
        texts[0],
        highlight_base,
        highlight_byte,
        round(msgs_rates[0] * 100, 2),
    )


def dec_fn(
    gen_model: str,
    text: str,
    msg_base: int,
    seed_scheme: str,
    window_length: int,
    private_key: int,
):
    model, tokenizer = ModelFactory.load_model(gen_model)
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
            gr.Dropdown(
                value=GlobalConfig.get("encrypt.default", "gen_model"),
                choices=ModelFactory.get_models_names(),
            ),
            gr.Textbox(),
            gr.Textbox(),
            gr.Number(int(GlobalConfig.get("encrypt.default", "start_pos"))),
            gr.Number(float(GlobalConfig.get("encrypt.default", "delta"))),
            gr.Number(int(GlobalConfig.get("encrypt.default", "msg_base"))),
            gr.Dropdown(
                value=GlobalConfig.get("encrypt.default", "seed_scheme"),
                choices=SeedSchemeFactory.get_schemes_name(),
            ),
            gr.Number(
                int(GlobalConfig.get("encrypt.default", "window_length"))
            ),
            gr.Number(int(GlobalConfig.get("encrypt.default", "private_key"))),
            gr.Number(bool(GlobalConfig.get("encrypt.default", "do_sample"))),
            gr.Number(
                float(
                    GlobalConfig.get("encrypt.default", "min_new_tokens_ratio")
                )
            ),
            gr.Number(
                float(
                    GlobalConfig.get("encrypt.default", "max_new_tokens_ratio")
                )
            ),
            gr.Number(int(GlobalConfig.get("encrypt.default", "num_beams"))),
            gr.Number(
                float(GlobalConfig.get("encrypt.default", "repetition_penalty"))
            ),
        ],
        outputs=[
            gr.Textbox(
                label="Text containing message",
                show_label=True,
                show_copy_button=True,
            ),
            gr.HighlightedText(
                label="Text containing message (Base highlighted)",
                combine_adjacent=False,
                show_legend=True,
                color_map={"correct": "green", "wrong": "red"},
            ),
            gr.HighlightedText(
                label="Text containing message (Byte highlighted)",
                combine_adjacent=False,
                show_legend=True,
                color_map={"correct": "green", "wrong": "red"},
            ),
            gr.Number(label="Percentage of message in text", show_label=True),
        ],
    )
    dec = gr.Interface(
        fn=dec_fn,
        inputs=[
            gr.Dropdown(
                value=GlobalConfig.get("decrypt.default", "gen_model"),
                choices=ModelFactory.get_models_names(),
            ),
            gr.Textbox(),
            gr.Number(int(GlobalConfig.get("decrypt.default", "msg_base"))),
            gr.Dropdown(
                value=GlobalConfig.get("decrypt.default", "seed_scheme"),
                choices=SeedSchemeFactory.get_schemes_name(),
            ),
            gr.Number(
                int(GlobalConfig.get("decrypt.default", "window_length"))
            ),
            gr.Number(int(GlobalConfig.get("decrypt.default", "private_key"))),
        ],
        outputs=[
            gr.Textbox(label="Message", show_label=True),
        ],
    )
    app = gr.TabbedInterface([enc, dec], ["Encrytion", "Decryption"])
    app.launch(share=True)
