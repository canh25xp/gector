from predict import predict, load
from huggingface_hub import hf_hub_download
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from json import dumps
from transformers import logging

logging.set_verbosity_error()


app = Flask(__name__)
api = Api(app)

roberta_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="roberta_1_gectorv2.th", cache_dir=".cache")
xlnet_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="xlnet_0_gectorv2.th", cache_dir=".cache")
bert_path = hf_hub_download(repo_id="canh25xp/GECToR-Roberta", filename="bert_0_gector.th", cache_dir=".cache")

print(f"roberta_path: {roberta_path}")
print(f"xlnet_path: {xlnet_path}")
print(f"bert_path: {bert_path}")
model_gector_roberta = load(str(roberta_path), "roberta")
model_gector_xlnet = load(str(xlnet_path), "xlnet")
model_gector_bert = load(str(bert_path), "bert")


class MODEL(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        model = json_data["model"]
        input = json_data["text_input_list"]

        print("================================================================================")
        print("Request:", dumps(json_data, indent=2, sort_keys=True))

        output = ""
        cnt_corrections = 0
        if model == "GECToR-Roberta":
            output, cnt_corrections = predict(input, model_gector_roberta)
        elif model == "GECToR-XLNet":
            output, cnt_corrections = predict(input, model_gector_xlnet)
        elif model == "GECToR-Bert":
            output, cnt_corrections = predict(input, model_gector_bert)
        else:
            output = "Model not supported"
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
