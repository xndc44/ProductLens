<h1>ProductLens</h1>

ProductLens is a Google Lens–style AI application that detects objects in images and provides context-aware, realistic answers using a combination of computer vision, embeddings, and large language models (LLMs).

Features

Zero-shot object detection using SAM (Segment Anything Model).

Semantic embeddings generated with DINOv2 for each detected object.

Retrieval-Augmented Generation (RAG) with a FAISS vector store and a curated knowledge base.

Context-aware question answering using LangChain-powered open-source LLMs.

FastAPI backend API for image uploads and inference.

Optional Streamlit frontend for quick testing and visualization.

Tech Stack

Computer Vision: PyTorch, SAM, DINOv2, PIL, OpenCV

NLP & LLM: LangChain, GPT4All (or other open-source LLMs)

Vector Database: FAISS for retrieval

Backend: FastAPI, Python 3.10+

Frontend (optional): Streamlit for quick prototyping

Other Tools: Docker, CUDA (GPU support), pickle caching
