from fastapi import FastAPI,File,UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import threading
import time
import subprocess
import uvicorn
#bot = ChatGPT()

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import PIL,glob

model = load_model('final_model_w_shuffle_2.h5')
#model.summary()

def predictor(x , model):
    #img = cv2.imread(x)
    img = Image.open(x)
    img = img.resize((48,48))
    imgar = np.array(img)
    kernel = np.array([[-0.1,-0.1,-0.1],[-0.1,2,-0.1],[-0.1,-0.1,-0.1]])
    imgar = cv2.filter2D(imgar,-1,kernel)



    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #img.resize((48,48,3))
    #img.reshape(48,48,3)
    #pred_val = model.predict(np.array([img])/255.0)
    pred_val = model.predict(np.array([imgar])/255.0)
    classification= pred_val[0,0]
    
    if classification >= 0.35 :
        op = "Depressed"
    else:
        op = "Not depressed"
    return [op,classification]



'''
#Imports for model 1
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
import neattext.functions as nfx
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')
model = keras.models.load_model('nlp_to_sucide.h5')
data=pd.read_csv('Suicide_Detection.csv')

def clean_text(text):
    text_length=[]
    cleaned_text=[]
    for sent in tqdm(text):
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        sent=nfx.remove_stopwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text,text_length

train_data,test_data=train_test_split(data,test_size=0.2,random_state=10)
cleaned_train_text,train_text_length=clean_text(train_data.text)
#cleaned_test_text,test_text_length=clean_text(test_data.text)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(cleaned_train_text)

# glove_embedding={}
with open('glove.840B.300d.pkl', 'rb') as fp:
    glove_embedding = pickle.load(fp)

v=len(tokenizer.word_index)

embedding_matrix=np.zeros((v+1,300), dtype=float)
for word,idx in tokenizer.word_index.items():
    embedding_vector=glove_embedding.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx]=embedding_vector

def detect_suicide(text):
    cleaned_text, length = clean_text(text)
    tokens = tokenizer.texts_to_sequences(cleaned_text)
    paded = pad_sequences(tokens,maxlen=40)
    x = model.predict([paded])
    if x >=0.50:
        print("contains suicidal thoughts",x)
    else:
        print("Doesn't contains suicidal thoughts",x)
#End of model 1
'''

IMAGEDIR = "fastapi-images/"

app = FastAPI()

@app.get("/{x}")
async def show(x:str):
    query = x
    #returned_output = subprocess.check_output("chatgpt what is the meaning of life?")
    returned_output =  subprocess.Popen("chatgpt "+query, shell=True, stdout=subprocess.PIPE).stdout.read()
    kl =  returned_output.decode("utf-8")
    #kl = "Hey Tarun.This is the response"
    return {"data":kl}

@app.post("/loki")
async def joke():
    return "okay it works"



@app.post("/images")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  

    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    valk = predictor(IMAGEDIR + file.filename , model)
    val = valk[0]
    thres = valk[1]

    return {"diagnosis": val,"value":str(thres)}


'''
@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
'''

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='10.20.206.33')


