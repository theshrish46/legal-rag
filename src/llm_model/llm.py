import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_llm_instance = None


def get_gemini_llm(temperature=0):
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=2048,
        )
        print("Gemini 3 Flash LLM initialized")

    return _llm_instance
