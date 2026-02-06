from .vectorstore import load_vectorstore

def retrieve_context(query: str):
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    return "\n".join([d.page_content for d in docs])
