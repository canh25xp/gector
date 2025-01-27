from gector.gec_model import GecBERTModel
from huggingface_hub import hf_hub_download
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from json import dumps
from transformers import logging

logging.set_verbosity_error()


def load(model_path, transformer_model):
    if transformer_model == "roberta":
        special_tokens_fix = 1
        min_error_prob = 0.50
        confidence_bias = 0.20
    elif transformer_model == "xlnet":
        special_tokens_fix = 0
        min_error_prob = 0.66
        confidence_bias = 0.35

    model = GecBERTModel(
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
    return model


def predict(lines, model, batch_size=32):
    test_data = [s.strip() for s in lines]
    predictions = []
    batch = []
    cnt_corrections = 0
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            batch = []
            cnt_corrections += cnt
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    # output = '<eos>'.join([' '.join(x) for x in predictions])
    output = [" ".join(x) for x in predictions]
    return output, cnt_corrections


app = Flask(__name__)
api = Api(app)

roberta_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="roberta_1_gectorv2.th", cache_dir=".cache")
xlnet_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="xlnet_0_gectorv2.th", cache_dir=".cache")

print(f"roberta_path: {roberta_path}")
print(f"xlnet_path: {xlnet_path}")
model_gector_roberta = load(str(roberta_path), "roberta")
model_gector_xlnet = load(str(xlnet_path), "xlnet")


class MODEL(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        model = json_data["model"]
        input = json_data["text_input_list"]

        print("================================================================================")
        print("Request:", dumps(json_data, indent=2, sort_keys=True))

        if model == "GECToR-Roberta":
            output, cnt_corrections = predict(input, model_gector_roberta)
        elif model == "GECToR-XLNet":
            output, cnt_corrections = predict(input, model_gector_xlnet)
        elif model == "T5-Large":
            output = ["Unsupported"]
        else:
            raise NotImplementedError(f"Model {model} is not recognized.")

        # fmt: off
        output_json = jsonify(
            {
                "model": model,
                "text_output_list": output
            }
        )
        # fmt: on

        print("Respond:", dumps(output_json.json, indent=4, sort_keys=True))
        print(f"Produced overall corrections: {cnt_corrections}")
        print("================================================================================")

        return output_json


api.add_resource(MODEL, "/components/model")


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=3000, use_reloader=False)
