import pandas as pd
import os
import time
import nest_asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper, llm_factory
from google import genai

from src.prompts.legal_templates import get_rag_chain
from dotenv import load_dotenv


load_dotenv()

nest_asyncio.apply()

# 1. Load data
df_test = pd.read_csv("tests/legal_eval_set.csv")
rag_chain = get_rag_chain()


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
ragas_judge = llm_factory("gemini-1.5-flash", provider="google", client=client)

# 2. Collect RAG responses (The Inference phase)
# Note: Inference uses your app's LLM, which also counts toward quota!
questions = df_test["question"].tolist()
ground_truths = df_test["ground_truth"].tolist()
answers = []
contexts = []

# Slicing to just 3 samples to save your daily quota
MAX_SAMPLES = 3
print(f"Began inference for {MAX_SAMPLES} samples...")

for i in range(MAX_SAMPLES):
    query = questions[i]
    print(f"Processing Inference {i+1}/{MAX_SAMPLES}...")
    response = rag_chain.invoke({"question": query, "chat_history": []})
    answers.append(response)
    raw_context = df_test[df_test["question"] == query]["context"].values[0]
    contexts.append([raw_context])

    # Wait to avoid hitting RPM limits during inference
    time.sleep(10)

# 3. Step-by-Step Evaluation (The Judge phase)
final_results = []
# We keep RunConfig strict even in the loop
run_config = RunConfig(max_workers=1, max_retries=10, timeout=120)

print("\nStarting RAGAS Scoring with High-Safety Cooldowns...")

for i in range(MAX_SAMPLES):
    print(f"--- Judging Sample {i+1}/{MAX_SAMPLES} ---")

    single_row = {
        "question": [questions[i]],
        "answer": [answers[i]],
        "contexts": [contexts[i]],
        "ground_truth": [ground_truths[i]],
    }
    mini_dataset = Dataset.from_dict(single_row)

    try:
        # Scoring one row can take ~4-5 API calls.
        # With 15 RPM, we should ideally wait after every row.
        res = evaluate(
            mini_dataset,
            metrics=[
                Faithfulness(llm=ragas_judge),
                AnswerRelevancy(llm=ragas_judge),
                ContextPrecision(llm=ragas_judge),
                ContextRecall(llm=ragas_judge),
            ],
            run_config=run_config,
        )
        final_results.append(res.to_pandas())
        print(f"✅ Success for sample {i+1}")
    except Exception as e:
        print(f"❌ Error on sample {i+1}: {e}")
        # If we hit a 429 here, wait even longer
        if "429" in str(e):
            print("Rate limit detected! Taking a 2-minute emergency break...")
            time.sleep(120)

    # MANDATORY COOLDOWN
    # Why 70s? Gemini's "Minute" is a sliding window.
    # Waiting 70s ensures the previous "burst" of 5 calls is fully cleared.
    if i < MAX_SAMPLES - 1:
        print("Cooldown: Waiting 80 seconds to reset API window...")
        time.sleep(80)

# 4. Merge and Save
if final_results:
    all_results_df = pd.concat(final_results, ignore_index=True)
    os.makedirs("tests", exist_ok=True)
    all_results_df.to_csv("tests/evaluation_results.csv", index=False)
    print("\n✅ Evaluation Complete! Results saved in tests/evaluation_results.csv")
else:
    print("\n❌ No samples were successfully evaluated.")
