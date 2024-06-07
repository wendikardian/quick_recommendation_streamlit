from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer
import json
import os
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("saved_tokenizer")

def tokenize_texts(texts, tokenizer, max_len=128):
    return tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
def save_tokens_to_json(tokens, file_path):
    with open(file_path, 'w') as f:
        json.dump(tokens, f)

ENDPOINT_ID="8785986311325483008"
PROJECT_ID="223463197253"


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "asia-southeast2",
    api_endpoint: str = "asia-southeast2-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}

    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    return predictions


def get_embeddings(texts, tokenizer, max_len= 128):
    tokens = tokenize_texts(texts, tokenizer, max_len)
    tokens = {key: value.numpy().tolist() for key, value in tokens.items()}
    instances = [{"input_ids": input_id, "attention_mask": attention_mask}
             for input_id, attention_mask in zip(tokens['input_ids'], tokens['attention_mask'])]

    data_to_save = {"instances": instances}
    with open('instances.json', 'w') as f:
        json.dump(data_to_save, f, indent=4)

    result = predict_custom_trained_model_sample(
        project=PROJECT_ID,
        endpoint_id=ENDPOINT_ID,
        location="asia-southeast2",
        instances=instances,
    )

    return result[0]


@app.get("/")
async def root():

    input_text = "menyukai bidang kesehatan dan ingin membantu orang lain dalam pemulihan fisik mereka. Jurusan ini mengkombinasikan ilmu pengetahuan tentang anatomi, fisiologi, dan terapi gerak untuk mengembangkan keterampilan dalam merawat pasien dengan berbagai kondisi fisik. Selain itu, fisioterapi juga melibatkan penggunaan teknologi canggih dan metode terapi yang inovatif untuk membantu pasien mencapai pemulihan yang optimal. Orang yang menyukai tantangan, memiliki empati, dan antusias dalam membantu orang lain akan merasa terpanggil untuk mengeksplorasi karir dalam bidang fisioterapi."
    input_embedding = get_embeddings([input_text], tokenizer,)
    print(input_embedding)
    embedding_array = np.array(input_embedding)

    # Get the indices of the top 5 highest values
    top_5_unique = np.argsort(embedding_array)[-5:][::-1]

    columns_to_consider = ['Matematika', 'Sains', 'Fisika', 'Sosiologi', 'Biologi', 'Kimia',
                            'Teknologi','Bisnis dan Ekonomi', 'Seni', 'Sastra dan Linguistik',
                        'Pendidikan', 'Hukum', 'Lingkungan', 'Kesehatan', 'Geografi',
                        'Komunikasi', 'Sejarah dan Filsafat']

    num_to_major = {i+1: columns_to_consider[i] for i in range(len(columns_to_consider))}

    # Map the top 5 unique numbers to their corresponding names
    top_5_majors_names = [num_to_major[num+1] for num in top_5_unique]
    return {"top_5_majors": top_5_majors_names}


uvicorn.run(app, host="127.0.0.1", port=8080)
