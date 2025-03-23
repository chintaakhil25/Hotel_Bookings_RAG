import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

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
    llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768")
    return llm

def infer_filter_type(query):
    """Infers a potential filter_type from the user's query."""
    query = query.lower()
    filter_mapping = {
        "cancellation": "cancellation_rate",
        "cancel": "cancellation_rate",  # Add variations
        "canceled": "cancellation_rate",
        "resort hotel": "cancellation_rate_by_hotel",
        "city hotel": "cancellation_rate_by_hotel",
        "revenue": "revenue_trend",
        "lead time": "lead_time",
        "country": "country_distribution",
        "market segment": "market_segment_distribution", #for market segment insights
        "distribution channel":"distribution_channel_distribution", #for distribution channel insights
        "repeated guests":"repeated_guest_percentage",
        "room type":"room_type_distribution",
        "booking changes":"avg_booking_changes",
        "deposit type":"deposit_type_distribution",
        "waiting list":"avg_waiting_days",
        "parking":"parking_percentage",
        "special requests":"avg_special_requests"
        # Add more mappings as needed
    }

    for keyword, filter_type in filter_mapping.items():
        if keyword in query:
            return filter_type
    return None  # No filter type inferred

def retrieve_insights(query, k, filter_type=None):
    """Retrieves insights from the FAISS index."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_insights", embedding_model)

    # --- Infer filter_type if not provided ---
    if filter_type is None:
        filter_type = infer_filter_type(query)
    # --- Rest of the retrieval logic remains the same ---

    if filter_type:
        documents = vector_store.docstore._dict.values()
        filtered_docs = []
        for doc in documents:
            if doc.metadata['type'] == filter_type:
                filtered_docs.append(doc)
        if filtered_docs:
            filtered_vs = FAISS.from_documents(filtered_docs, embedding_model)
            results = filtered_vs.similarity_search_with_score(query, k=k)
        else:
            results = []
    else:
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