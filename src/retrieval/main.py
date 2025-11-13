from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

class Response(BaseModel):
    results: list

@app.post("/retrieve", response_model=Response)
async def retrieve(query: Query):
    # TODO: Implement hybrid search logic
    return {"results": []}
