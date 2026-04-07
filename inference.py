from fastapi import FastAPI
import torch
import gymnasium as gym

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return {"observation": [0.0], "info": {}}

@app.post("/step")
def step(action: dict):
    return {"observation": [0.0], "reward": 0.0, "terminated": True, "truncated": False, "info": {}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
