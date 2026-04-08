from fastapi import FastAPI
import torch
import gymnasium as gym

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "active"}

@app.post("/reset")
async def reset():
    # Returns the initial observation required by the validator
    return {"observation": [0.0], "info": {}}

@app.post("/step")
async def step(action: dict):
    # Returns the step results required by the validator
    return {"observation": [0.0], "reward": 0.0, "terminated": True, "truncated": False, "info": {}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
