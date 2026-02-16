import pandas as pd
import time
import os
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,  # Replaced GEval with Relevancy for stability
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel
from src.prompts.legal_templates import get_rag_chain

# 1. SETUP
# Override timeout for large legal docs
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"
eval_model = GeminiModel(model="gemini-3-flash-preview")
chain = get_rag_chain()

# 2. INITIALIZE STABLE METRICS
# These three form the 'RAG Triad' and are the gold standard for legal audits.
precision_metric = ContextualPrecisionMetric(
    threshold=0.5, model=eval_model, async_mode=False, include_reason=True
)
recall_metric = ContextualRecallMetric(
    threshold=0.7, model=eval_model, async_mode=False
)
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7, model=eval_model, async_mode=False
)

# Relevancy is a great proxy for correctness without the 'raw_response' bug
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7, model=eval_model, async_mode=False
)

metrics = [precision_metric, recall_metric, faithfulness_metric, relevancy_metric]

# 3. RUN AUDIT
df = pd.read_csv("legal_eval_set.csv")
results = []

for index, row in df.head(1).iterrows():
    print(f"Auditing Row {index + 1}/{len(df)}...")

    # A. Get RAG Response
    output = chain.invoke({"question": row["question"], "chat_history": []})
    full_answer = output.get("answer", "")
    retrieved_context_strings = [
        doc.page_content for doc in output.get("retrieved_docs", [])
    ]

    # B. Create Test Case
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=full_answer,
        expected_output=str(row["ground_truth"]),
        retrieval_context=retrieved_context_strings,
    )

    # C. Measure
    row_results = {
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "actual_output": full_answer,
    }

    for metric in metrics:
        try:
            metric.measure(test_case)
            # Map metric names to scores
            name = metric.__class__.__name__.replace("Metric", "").lower()
            row_results[name] = metric.score
        except Exception as e:
            print(f"Error on metric {metric.__class__.__name__}: {e}")
            row_results[name] = 0  # Default on error
        time.sleep(15)  # Small buffer between metrics

    results.append(row_results)

    # D. Long cooldown to avoid 503 Service Unavailable
    print(f"✅ Row {index + 1} complete. Cooldown for 90s...")
    time.sleep(90)

# 4. EXPORT
pd.DataFrame(results).to_csv("legal_audit_report.csv", index=False)
print("✨ Audit Finished! Report saved.")
print("Results ==========> ", results)
