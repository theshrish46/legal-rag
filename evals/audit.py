import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your engine
from src.rag_engine import process_query

# 1. Load the Test Data (Generated in previous step)
csv_path = os.path.join(current_dir, "generated_data.csv")

if not os.path.exists(csv_path):
    print("❌ Error: generated_data.csv not found. Run generate_dataset.py first.")
    exit()

print("Loading test data...")
df = pd.read_csv(csv_path)

# 2. Prepare RAGAS Data Container
results = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

# 3. Run the RAG Pipeline
print(f"Running evaluation on {len(df)} questions...")

for index, row in df.iterrows():
    question = row["question"]
    ground_truth = row["ground_truth"]

    # Call your RAG Engine
    print(f"Processing Q{index+1}: {question[:30]}...")
    output = process_query(question)

    results["question"].append(question)
    results["answer"].append(output["answer"])
    results["contexts"].append(output["contexts"])
    results["ground_truth"].append(ground_truth)

# 4. Configure Judge LLM (Use Gemini to save money)
# RAGAS needs an LLM to judge if the answer is faithful.
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-1.5-flash"))

# 5. Calculate Scores
print("Calculating RAGAS metrics...")
dataset = Dataset.from_dict(results)

scores = evaluate(
    dataset=dataset,
    metrics=[faithfulness(), answer_relevancy()],
    llm=evaluator_llm,  # Use Gemini as the judge
)

# 6. Save Results
print("\n🎉 Evaluation Complete!")
print(scores)

# Save detailed results to CSV
df_results = scores.to_pandas()
df_results.to_csv(os.path.join(current_dir, "final_audit_report.csv"), index=False)
