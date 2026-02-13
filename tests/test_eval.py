import pandas as pd
import time
import os
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)

# Fixed Import: LLMTestCaseParams must be imported from deepeval.test_case
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel
from src.prompts.legal_templates import get_rag_chain

# 1. SETUP MODELS & CHAIN
# Ensure your GOOGLE_API_KEY is in your environment variables
eval_model = GeminiModel(model="gemini-2.5-flash")
chain = get_rag_chain()

# 2. INITIALIZE METRICS
precision_metric = ContextualPrecisionMetric(threshold=0.7, model=eval_model)
recall_metric = ContextualRecallMetric(threshold=0.7, model=eval_model)
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=eval_model)

# Fixed GEval: Added name, criteria, and evaluation_params correctly
correctness_metric = GEval(
    name="Legal Correctness",
    model=eval_model,
    criteria="Assess whether the actual output matches the expected output in terms of legal facts and numbers.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

metrics = [precision_metric, recall_metric, faithfulness_metric, correctness_metric]

# 3. RUN AUDIT
df = pd.read_csv("legal_eval_set.csv")
results = []

print(f"🚀 Starting audit for {len(df)} legal queries...")

for index, row in df.iterrows():
    print(f"Auditing Row {index + 1}/{len(df)}...")

    # A. Get RAG Response
    # Note: Ensure your chain returns 'answer' and 'retrieved_docs'
    output = chain.invoke({"question": row["question"], "chat_history": []})

    full_answer = output.get("answer", "")
    # Convert LangChain Document objects to raw strings for DeepEval
    docs = output.get("retrieved_docs", [])
    retrieved_context_strings = [doc.page_content for doc in docs]

    # B. Create Test Case (MOVED INSIDE THE LOOP)
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=full_answer,
        expected_output=str(row["ground_truth"]),
        retrieval_context=retrieved_context_strings,
    )

    # C. Measure each metric
    for metric in metrics:
        metric.measure(test_case)
        # Small sleep between metrics to avoid rate limits
        time.sleep(2)

    # D. Store results
    results.append(
        {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "actual_output": full_answer,
            "faithfulness": faithfulness_metric.score,
            "precision": precision_metric.score,
            "recall": recall_metric.score,
            "correctness": correctness_metric.score,
            "reasoning": correctness_metric.reason,
        }
    )

    # E. Long sleep to stay within Gemini Free Tier limits (approx 15 RPM)
    print(f"✅ Row {index + 1} complete. Sleeping for cooldown...")
    time.sleep(90)

# 4. EXPORT
report_name = "legal_audit_report.csv"
pd.DataFrame(results).to_csv(report_name, index=False)
print(f"✨ Audit Finished! Report saved to {report_name}")
print(f"Results ===============> {results}")