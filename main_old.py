from fastapi import FastAPI
import threading
import time
import subprocess

#bot = ChatGPT()

app = FastAPI()

@app.get("/{x}")
async def show(x:str):
    # returns output as byte string
    query = x
    #returned_output = subprocess.check_output("chatgpt what is the meaning of life?")
    returned_output =  subprocess.Popen("chatgpt "+query, shell=True, stdout=subprocess.PIPE).stdout.read()
    # using decode() function to convert byte string to string
    kl =  returned_output.decode("utf-8")
    return {"Criii":kl}




