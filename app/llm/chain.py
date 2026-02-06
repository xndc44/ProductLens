from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate

llm = GPT4All(model="gpt4all-falcon-q4.bin")

prompt = PromptTemplate(
    input_variables=["objects", "context", "question"],
    template="""
You are ProductLens, an AI that understands images.

Detected objects:
{objects}

Context:
{context}

Question:
{question}

Answer clearly and realistically.
"""
)

def run_llm(objects, context, question):
    return llm(
        prompt.format(
            objects=objects,
            context=context,
            question=question
        )
    )
