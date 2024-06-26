import streamlit as st
from transformers import AutoTokenizer
import json
import os
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import numpy as np
import toml

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

service_account_info = st.secrets["gcp_service_account"]
print(service_account_info)
# # Set the environment variable to point to the service account file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'compassion-ly-app-0267ce352c33.json'

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_info['private_key']

tokenizer = AutoTokenizer.from_pretrained("saved_tokenizer")

def tokenize_texts(texts, tokenizer, max_len=256):
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
    predictions = response.predictions
    return predictions

def get_embeddings(texts, tokenizer, max_len= 256):
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


st.title("Major Recommendation System")
input_text = st.text_area("Enter a description of your interests and career aspirations:")

if st.button("Get Recommendations"):
    if len(input_text) < 70:
        st.error("Please enter at least 70 characters.")
    else:
        input_embedding = get_embeddings([input_text], tokenizer)
        embedding_array = np.array(input_embedding)

        top_5_unique = np.argsort(embedding_array)[-5:][::-1]

        columns_to_consider = ['Matematika', 'Sains', 'Fisika', 'Sosiologi', 'Biologi', 'Kimia',
                               'Teknologi', 'Bisnis dan Ekonomi', 'Seni', 'Sastra dan Linguistik',
                               'Pendidikan', 'Hukum', 'Lingkungan', 'Kesehatan', 'Geografi',
                               'Komunikasi', 'Sejarah dan Filsafat']

        num_to_major = {i+1: columns_to_consider[i] for i in range(len(columns_to_consider))}

        top_5_majors_names = [num_to_major[num+1] for num in top_5_unique]
        st.write("Top 5 recommended majors for you:")
        for i, major in enumerate(top_5_majors_names, 1):
            st.write(f"{i}. {major}")
