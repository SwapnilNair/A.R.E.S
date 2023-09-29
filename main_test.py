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

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='10.20.206.33')


