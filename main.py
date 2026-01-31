from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="GGUF CPU Inference Service")

MODEL_PATH = "/home/thenameisvicky/Documents/quant_neural_models/rocket_3B_Q4.gguf"

print("Loading GGUF model...")
llm = Llama(model_path=MODEL_PATH)

class Prompt(BaseModel):
    prompt: str

@app.post("/infer")
def infer(data: Prompt):
    output = llm(data.prompt, max_tokens=100, temperature=0.7)
    return {"response": output['choices'][0]['text']}
    