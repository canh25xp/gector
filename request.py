import requests
from json import dumps

response = requests.post(
    "http://localhost:3000/components/model",
    json={
        "model": "GECToR-Roberta",
        "text_input_list": ["He do this", "I is a engineer", "The quick brown fox jump over the lazy dog"],
    },
).json()

data = response["text_output_list"]

print(dumps(response, indent=4, sort_keys=True))
