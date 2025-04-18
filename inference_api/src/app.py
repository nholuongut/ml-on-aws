import fastapi
from fastapi import Depends
from fastapi.responses import PlainTextResponse,JSONResponse
import os
import random
import pickle

import logging
import boto3
from botocore.exceptions import ClientError

from mlem.api import load
from tensorflow.keras.preprocessing.sequence import pad_sequences

router = fastapi.APIRouter()

s3_client = boto3.client('s3')
model_bucket_name = os.environ.get('MODEL_S3_BUCKET_NAME')

# download models
s3_client.download_file(model_bucket_name, 'models/tf', 'model/tf')
s3_client.download_file(model_bucket_name, 'models/tf.mlem', 'model/tf.mlem')

s3_client.download_file(model_bucket_name, 'encoder/tf', 'encoder/tf')
s3_client.download_file(model_bucket_name, 'encoder/tf.mlem', 'encoder/tf.mlem')

s3_client.download_file(model_bucket_name, 'tokenizer.pickle', 'tokenizer/tokenizer.pickle')

max_sequence_len=122
model = load("model/tf")
encoder=load("encoder/tf")
loaded_tokenizer=None
with open('tokenizer/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


@router.get('/ping')
async def pingmethod():
    return JSONResponse(status_code=200, content={"status":"working"})

@router.post('/parseinputs',response_class=JSONResponse)
async def parsemethod(payload: dict):
    status_code=500
    resp={"status":"error"}
    try:
        payloadinput=payload.copy()
        input_text = payloadinput['text']
        print(input_text)
        input_seq = loaded_tokenizer.texts_to_sequences([input_text])
        input_features = pad_sequences(input_seq, maxlen = max_sequence_len, padding = 'post')
        probs = model.predict(input_features)
        predicted_y = probs.argmax(axis=-1)
        print(encoder.classes_[predicted_y][0])
        resp={"sentiment":encoder.classes_[predicted_y][0]}
        status_code=200
    except Exception as e:
        print(e)

    print(payload)
    return JSONResponse(status_code=status_code, content=resp)