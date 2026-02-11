import os
import pandas as pd
import time
import nest_asyncio
from dotenv import load_dotenv
from datasets import Dataset

# 1. Imports: Metrics must come from .collections in v0.4
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics.collections import Faithfulness, ContextRecall, FactualCorrectness
from ragas.llms import llm_factory
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

# Your project imports
from src.prompts.legal_templates import get_rag_chain

load_dotenv()
nest_asyncio.apply()

# 1. Setup Student (RAG Chain)
rag_chain = get_rag_chain()

# 2. Setup Judge (Unified Factory)
# We use the ChatGoogleGenerativeAI object directly in the factory
# client = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro", google_api_key=os.environ.get("GOOGLE_API_KEY")
# )

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# In v0.4, llm_factory returns the correctly initialized Ragas LLM
evaluator_llm = llm_factory(model="gemini-1.5-pro", provider="google", client=client)

# 3. Prepare Dataset
df_test = pd.read_csv("tests/legal_eval_set.csv")
MAX_SAMPLES = 3
data = []

print(f"Running inference for {MAX_SAMPLES} samples...")

for i in range(MAX_SAMPLES):
    question = df_test["question"].iloc[i]
    ground_truth = df_test["ground_truth"].iloc[i]

    # Get response from your chain
    response = rag_chain.invoke({"question": question, "chat_history": []})
    context = [df_test["context"].iloc[i]]

    data.append(
        {
            "user_input": question,
            "response": response,
            "retrieved_contexts": context,
            "reference": ground_truth,
        }
    )
    print(f"Sample {i+1} inference complete. Waiting 10s...")
    time.sleep(80)

eval_dataset = EvaluationDataset.from_list(data)

# 4. Initialize Metrics (The Fix)
# Do NOT pass the LLM inside these parentheses.
# This is what causes the "All metrics must be initialised" error.
metrics = [
    Faithfulness(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm, mode="precision"),
    ContextRecall(llm=evaluator_llm),
]

print("\nStarting Evaluation Phase...")

try:
    # We pass the judge (evaluator_llm) ONCE here.
    # Ragas will automatically inject it into the metrics above.
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=RunConfig(timeout=240, max_retries=5, max_workers=1),
    )

    # 5. Save results
    df_results = results.to_pandas()
    df_results.to_csv("tests/evaluation_results.csv", index=False)
    print("\n✅ Success! CSV saved to tests/evaluation_results.csv")
    print(df_results.mean(numeric_only=True))

except Exception as e:
    print(f"❌ Eval failed: {e}")
