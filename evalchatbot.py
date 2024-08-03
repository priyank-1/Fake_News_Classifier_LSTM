import os
import mlflow
import pandas as pd
import streamlit as st
import warnings
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Load environment variables
load_dotenv()
# Disable FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load GROQ and Google API keys from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-it")

# Prompt Template
prompt_template = """
Answer the questions based on the provided conttext only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Document embedding and vector store setup
def vector_embedding():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./indiacensus")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# User input for questions
prompt1 = st.text_input("What do you want to ask from the documents?")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready")

import time

# Q&A Chatbot function
def chatbot_response(prompt1):
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)   
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        start = time.process_time()
        response = retrieval_chain({"input": prompt1})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['result'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["source_documents"]):
                st.write(doc.page_content)
                st.write("----------------------------------")
        return response['result']
    return ""

# Evaluate the model using cosine similarity
def evaluate_model_with_cosine_similarity(eval_df):
    generated_answers = []
    for question in eval_df["questions"]:
        generated_answers.append(chatbot_response(question))
    
    ground_truths = eval_df["ground_truth"].tolist()
    
    # Get embeddings for both generated answers and ground truths
    generated_embeddings = [embeddings.embed_query(answer) for answer in generated_answers]
    ground_truth_embeddings = [embeddings.embed_query(truth) for truth in ground_truths]
    
    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(generated_embeddings, ground_truth_embeddings)
    
    return cosine_sim

# Cosine similarity computation function
def compute_cosine_similarity(embeddings1, embeddings2):
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        similarities.append(cosine_similarity(np.array(emb1).reshape(1, -1), np.array(emb2).reshape(1, -1))[0][0])
    return np.mean(similarities)

# Evaluation DataFrame
eval_df = pd.DataFrame({
    "questions": [
        "What is MLflow?",
        "What is Spark?",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",
        "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, offering improvements in speed and ease of use. Spark provides libraries for various tasks such as data ingestion, processing, and analysis through its components like Spark SQL for structured data, Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
    ],
})

# Evaluate the model and display the cosine similarity
cosine_sim = evaluate_model_with_cosine_similarity(eval_df)
st.write(f"Cosine Similarity: {cosine_sim}")
