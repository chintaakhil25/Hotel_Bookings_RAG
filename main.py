from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from rag_system import (load_data_and_create_documents, create_vector_store,
                         load_vector_store, load_llm, retrieve_insights,
                         generate_answer)

app = FastAPI()

# Load the RAG components on startup
if os.path.exists("faiss_insights"):
    vector_store = load_vector_store()
else:
    documents = load_data_and_create_documents()
    vector_store = create_vector_store(documents)
llm = load_llm()

class Query(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hotel Booking Insights RAG API!"}

@app.post("/ask")
def ask_question(query: Query):
    """Answers a question using the RAG system."""
    try:
        print(f"Received query: {query.text}")
        
        # Retrieve insights, using the inferred filter_type
        results = retrieve_insights(query.text, k=3)
        print(f"Retrieved results: {results}")

        context = ""
        for doc, score in results:
            context += f"Source (Type: {doc.metadata['type']}): {doc.page_content}\n"
        print(f"Generated context: {context}")

        answer = generate_answer(llm, query.text, context)
        print(f"Generated answer: {answer}")

        return {"question": query.text, "answer": answer, "context": context}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)