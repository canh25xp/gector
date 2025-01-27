from predict import predict, load
from huggingface_hub import hf_hub_download
import gradio as gr
import difflib
from transformers import logging

logging.set_verbosity_error()


def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    differ = difflib.Differ()
    diff = list(differ.compare(original_words, corrected_words))
    highlighted = []
    for word in diff:
        if word.startswith("+ "):
            highlighted.append(f"<b><span style='color:green'>{word[2:]}</span></b>")
        elif word.startswith("- "):
            highlighted.append(f"<b><span style='color:red'>{word[2:]}</span></b>")
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

model_select = gr.Dropdown(["GECToR-Roberta", "GECToR-XLNet", "GECToR-Bert", "Roberta-Large"], label="Select model")

output_text = gr.HTML(label="Output text")

examples = [
    [
        "I seen her at the store yesterday",  # I saw her at the store yesterday
        "GECToR-Roberta",
    ],
    [
        "We was going to the park when it started to raining",  # We were going to the park when it started to rain
        "GECToR-XLNet",
    ],
    [
        "She don't like to eat broccoli because they are green",  # She doesn't like to eat broccoli because it is green
        "GECToR-Bert",
    ],
    [
        "He are one of the best players in the team",  # He is one of the best players on the team
        "Roberta-Large",
    ],
]


if __name__ == "__main__":
    roberta_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="roberta_1_gectorv2.th", cache_dir=".cache")
    xlnet_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="xlnet_0_gectorv2.th", cache_dir=".cache")
    bert_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="bert_0_gector.th", cache_dir=".cache")

    print(f"roberta_path: {roberta_path}")
    print(f"xlnet_path: {xlnet_path}")
    print(f"bert_path: {bert_path}")
    model_gector_roberta = load(str(roberta_path), "roberta")
    model_gector_xlnet = load(str(xlnet_path), "xlnet")
    model_gector_bert = load(str(bert_path), "bert")

    def get_prediction(text, model, highlight):
        output = ""
        cnt_corrections = 0
        if model == "GECToR-Roberta":
            output, cnt_corrections = predict([text], model_gector_roberta)
        elif model == "GECToR-XLNet":
            output, cnt_corrections = predict([text], model_gector_xlnet)
        elif model == "GECToR-Bert":
            output, cnt_corrections = predict([text], model_gector_bert)
        else:
            output = "Model not supported"

        output = "\n".join(output)

        print(f"Produced overall corrections: {cnt_corrections}")

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
