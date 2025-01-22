from gector.gec_model import GecBERTModel
from huggingface_hub import hf_hub_download
import gradio as gr
import difflib
from transformers import logging

logging.set_verbosity_error()


def load(model_path, transformer_model):
    special_tokens_fix = 1
    min_error_prob = 0.50
    confidence_bias = 0.20

    return GecBERTModel(
        vocab_path="test_fixtures/roberta_model/vocabulary",
        model_paths=[model_path],
        max_len=50,
        min_len=3,
        iterations=5,
        min_error_probability=min_error_prob,
        lowercase_tokens=False,
        model_name=transformer_model,
        special_tokens_fix=special_tokens_fix,
        log=False,
        confidence=confidence_bias,
    )


def predict(lines, model, batch_size=32):
    test_data = [s.strip() for s in lines]  # Remove trailling spaces
    predictions = []
    batch = []
    cnt_corrections = 0
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    # output = '<eos>'.join([' '.join(x) for x in predictions])
    output = [" ".join(x) for x in predictions]
    return "\n".join(output)


def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    differ = difflib.Differ()
    diff = list(differ.compare(original_words, corrected_words))
    highlighted = []
    for word in diff:
        if word.startswith("+ "):
            highlighted.append(
                f"<b><span style='color:green'>{word[2:]}</span></b>"
            )
        elif word.startswith("- "):
            highlighted.append(
                f"<b><span style='color:red'>{word[2:]}</span></b>"
            )
        elif word.startswith("? "):
            continue
        else:
            highlighted.append(word[2:])
    return " ".join(highlighted)


# Gradio interface
title = "Gector web interface"

description = "Enter a text and select a model to correct grammar errors."

text_input = gr.Textbox(lines=5, label="Input text")

check_box = gr.Checkbox(label="Highlight output")

model_select = gr.Dropdown(
    ["Roberta", "Roberta-Large", "Bert"], label="Select model"
)

output_text = gr.HTML(label="Output text")

examples = [
    [
        "I seen her at the store yesterday",  # I saw her at the store yesterday
        "Roberta",
    ],
    [
        "We was going to the park when it started to raining",  # We were going to the park when it started to rain
        "Roberta-Large",
    ],
    [
        "She don't like to eat broccoli because they are green",  # She doesn't like to eat broccoli because it is green
        "Roberta-Large",
    ],
    [
        "He are one of the best players in the team",  # He is one of the best players on the team
        "Roberta-Large",
    ],
]


if __name__ == "__main__":
    roberta_path = hf_hub_download(
        repo_id="canh25xp/GECToR-Roberta",
        filename="roberta_1_gectorv2.th",
        cache_dir=".cache",
    )
    # roberta_large_path = hf_hub_download("canh25xp/GECToR-Roberta", "roberta-large_1_pie_1bw_st3.th")
    # roberta_large_spell_path = hf_hub_download("canh25xp/GECToR-Roberta", "roberta-large-spell50k.th")
    # bert_path = hf_hub_download("canh25xp/GECToR-Roberta", "bert_0_gector.th")

    print(f"roberta_path: {roberta_path}")

    roberta = load(str(roberta_path), "roberta")

    def get_prediction(text, model_name, highlight):
        output = ""
        if model_name == "Roberta":
            output = predict([text], roberta)
        if model_name == "Roberta-Large":
            output = "Not supported yet"
        if model_name == "Bert":
            output = "Not supported yet"

        if highlight:
            return highlight_differences(text, output)
        else:
            return output

    app = gr.Interface(
        fn=get_prediction,
        inputs=[text_input, model_select, check_box],
        outputs=output_text,
        title=title,
        description=description,
        examples=examples,
        allow_flagging="never",
    )

    app.launch(share=False)
