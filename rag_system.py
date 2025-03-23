import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

def load_data_and_create_documents(csv_path='insights_df.csv'):
    insights_df = pd.read_csv(csv_path)
    documents = []
    for i, row in insights_df.iterrows():
        doc = Document(page_content=row['insight'], metadata={"type": row['type']})
        documents.append(doc)
    return documents

def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local("faiss_insights")
    return vector_store

def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_insights", embedding_model, allow_dangerous_deserialization=True)

def load_llm():
    llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile",api_key = api_key)
    return llm




def retrieve_insights(query, k):
    """Retrieves insights from the FAISS index."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_insights", embedding_model,allow_dangerous_deserialization=True)
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

def generate_answer(llm, query, context):
    """Generates an answer using the LLM and context."""
    prompt = f"""
    You are a helpful assistant that answers questions based on provided context.
    If the answer is not in the context, respond with "I don't have enough information to answer."

    Context:
    {context}

    Question: {query}

    Answer:
    """
    answer = llm.invoke(prompt)
    return answer.content

