import pandas as pd
from src.database import get_vector_store
from src.llm_model.llm import get_gemini_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import random
import time


def generate_test_set(num_samples=10):
    vector_store = get_vector_store()
    llm = get_gemini_llm()

    # 1. Grab random chunks from your Qdrant DB to use as context
    # Note: Using a simple search to get diverse chunks
    random_docs = vector_store.similarity_search("agreement", k=num_samples)

    gen_prompt = ChatPromptTemplate.from_template(
        """
    You are a Senior Legal Auditor creating a "Gold Standard" test set for a RAG system.
    
    [CONTEXT]: {context}
    
    TASK: Generate one complex Question and its corresponding Ground Truth Answer.
    The question should require careful reading of the context. 
    Use one of these styles:
    1. "Negative Constraint": What is specifically PROHIBITED or NOT included?
    2. "Numerical/Temporal": Questions about deadlines, percentages, or dollar amounts.
    3. "Conditional": "What happens IF X occurs?"
    
    Return ONLY a JSON object:
    {{
        "question": "...",
        "ground_truth": "..."
    }}
"""
    )

    chain = gen_prompt | llm | JsonOutputParser()

    dataset = []
    for i, doc in enumerate(random_docs):
        print(f"[{i+1}/{len(random_docs)}] Generating Q&A for chunk...")

        try:
            time.sleep(2)
            res = chain.invoke({"context": doc.page_content})
            res["context"] = doc.page_content
            dataset.append(res)

            print("Sleeping to avoid rate limits...")
            time.sleep(60)

        except Exception as e:
            if "429" in str(e):  # 429 is the standard 'Rate Limit Exceeded' error
                print("Rate limit hit! Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                print(f"Error: {e}")

    # 2. Save to CSV for RAGAS
    df = pd.DataFrame(dataset)
    df = df.rename(columns={"question": "user_input", "ground_truth": "reference"})
    df["retrieved_contexts"] = df["context"].apply(lambda x: [x])
    df.to_csv("tests/legal_eval_set.csv", index=False)
    return df


if __name__ == "__main__":
    generate_test_set(5)  # Start small
