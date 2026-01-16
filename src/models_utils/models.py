from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from langchain_google_genai import ChatGoogleGenerativeAI


def get_embedding_model():
    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
