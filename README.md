# Hotel Booking Insights: An LLM-Powered Q&A System

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask natural language questions about a hotel booking dataset and receive insightful answers. The system leverages a combination of:

*   **Data Preprocessing and Analysis:**  Exploratory data analysis and pre-computation of key insights using Pandas, NumPy, Matplotlib, and Seaborn.
*   **Vector Database:**  FAISS (Facebook AI Similarity Search) for efficient storage and retrieval of precomputed insights.
*   **Large Language Model (LLM):** Groq's cloud-based LLMs (specifically `llama-3.3-70b-versatile`) for natural language understanding and generation, accessed via LangChain.
*   **REST API:**  A FastAPI-based REST API to expose the Q&A functionality to external applications.

The system is designed to be modular, with separate components for data preprocessing, RAG logic, and the API, making it easy to understand, maintain, and extend.

## Repository Structure

├── data/ <- (Optional) Raw dataset.
│ └── hotel_bookings.csv,insights_df.csv(from data_preprocessing.ipynb)
├── notebooks/ <- Colab notebooks for data analysis and RAG demonstration.
│ ├── 1_Data_Preprocessing_and_Analysis.ipynb <- Data loading, cleaning, EDA, insight generation.
│ └── 2_RAG_Implementation.ipynb <- Demonstrates RAG system using rag_system.py.
├── rag_system.py <- Core RAG logic (functions for retrieval and generation).
├── main.py <- FastAPI application (REST API).
├── requirements.txt <- Python dependencies.
├── faiss_insights/ <- Directory containing the saved FAISS index (created by 2_RAG_Implementation.ipynb).
└── README.md <- This file.
|__ .env <- for loading api key

## Setup Instructions

These instructions assume you have Python 3.8+ and `pip` installed.

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_name>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Obtain a Groq API Key:**

    *   Create an account on the Groq Cloud platform: [https://console.groq.com/](https://console.groq.com/)
    *   Obtain an API key.
    *   Set the API key as an environment variable named `GROQ_API_KEY`.  You can do this in your terminal:

        ```bash
        export GROQ_API_KEY="your_actual_groq_api_key"  # macOS/Linux
        set GROQ_API_KEY=your_actual_groq_api_key  # Windows
        ```
    *   Alternatively, you can add these lines at the top of your code
        ```python
        from dotenv import load_dotenv
        load_dotenv()
        ```
        and `pip install python-dotenv` and add to `requirements.txt`. And create a `.env` file and add `GROQ_API_KEY = "your_api_key"`
    *   **Important:**  Do *not* hardcode your API key directly into your code.

5.  **Download the Dataset (If Necessary):**

    *   If the `data/hotel_bookings.csv` file is *not* included in the repository (due to size or licensing restrictions), download the dataset from Kaggle: [https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
    *   Place the downloaded `hotel_bookings.csv` file in the `data/` directory.

6. **Generate Insights and Build FAISS Index:**
   * Open `notebooks/1_Data_Preprocessing_and_Analysis.ipynb` in VS Code and run all cells. This will generate `insights_df.csv` in the project root.
   * Open `notebooks/2_RAG_Implementation.ipynb` in VS Code. Run all cells. This will create the `faiss_insights` directory, containing the FAISS index.

7.  **Run the API:**

    *   Open a terminal in your project's root directory (where `main.py` is located).
    *   Make sure your virtual environment is activated.
    *   Run the following command:

        ```bash
        uvicorn main:app --reload
        ```

    *   The API will be accessible at `http://127.0.0.1:8000`.

## API Documentation

The API provides the following endpoints:

*   **`/` (GET):**  Returns a welcome message.
*   **`/ask` (POST):**  Answers a question about the hotel booking data.

**`/ask` Endpoint:**

*   **Request Body:**  A JSON object with the following format:

    ```json
    {
        "text": "Your question here"
    }
    ```

    *   `text`:  The user's natural language question (required).

*   **Response Body:** A JSON object with the following format:

    ```json
    {
        "question": "The user's question",
        "answer": "The LLM-generated answer",
        "context": "The retrieved insights used to generate the answer"
    }
    ```

**Example using `Swagger UI`:**


** Provide your query
{"text": "What is the overall cancellation rate?"}' at http://127.0.0.1:8000/docs#/default/ask_question_ask_post



