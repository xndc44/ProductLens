from fastapi import FastAPI, UploadFile
from PIL import Image
from app.vision.pipeline import run_vision_pipeline
from app.rag.retriever import retrieve_context
from app.llm.chain import run_llm

app = FastAPI()

@app.post("/analyze")
async def analyze(image: UploadFile, question: str):
    img = Image.open(image.file).convert("RGB")

    objects = run_vision_pipeline(img)
    object_summary = ", ".join([o["id"] for o in objects])

    context = retrieve_context(question)
    answer = run_llm(object_summary, context, question)

    return {
        "objects": objects,
        "answer": answer
    }
