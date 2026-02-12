import os
import pandas as pd
import asyncio
import nest_asyncio
import multiprocessing  # <--- Add this
from dotenv import load_dotenv
from ragas import EvaluationDataset, RunConfig, experiment
from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from ragas.backends import LocalJSONLBackend
from google import genai

# Your project imports
from src.prompts.legal_templates import get_rag_chain
from google import genai


def setup_env():
    load_dotenv()
    nest_asyncio.apply()

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    evaluator_llm = llm_factory(
        model="gemini-1.5-pro", provider="google", client=client
    )
    return evaluator_llm, get_rag_chain()


evaluator_llm, rag_chain = setup_env()
faithfulness = Faithfulness(llm=evaluator_llm)


@experiment()
async def agent_eval(row, **kwargs):
    try:
        u_input = str(row.user_input)

        agent_output = rag_chain.invoke({"question": u_input, "chat_history": []})

        # 2. Extract answer text safely
        response_text = (
            agent_output.get("answer")
            if isinstance(agent_output, dict)
            else str(agent_output)
        )

        print(f"Agent generated response. Sleeping 85s before scoring...")
        await asyncio.sleep(85)

        # scoring_task = faithfulness.ascore(
        #     user_input=u_input,
        #     response=str(response_text),
        #     retrieved_contexts=list(row.retrieved_contexts),
        # )

        # result = await scoring_task

        # if asyncio.iscoroutine(result):
        #     result = await result
        # final_score = result.value if hasattr(result, "value") else float(result)

        def calculate_score():
            return faithfulness.score(
                user_input=u_input,
                response=str(response_text),
                retrieved_contexts=list(row.retrieved_contexts),
            )

        # We 'await' the result of the thread
        f_score = await asyncio.to_thread(calculate_score)

        return {
            "user_input": u_input,
            "response": str(response_text),
            "faithfulness": f_score.value if hasattr(f_score, "value") else f_score,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


async def main():
    # 1. Load and Clean Data INSIDE main
    df = pd.read_csv("tests/legal_eval_set.csv")
    df = df.rename(columns={"question": "user_input", "ground_truth": "reference"})
    df["retrieved_contexts"] = df["context"].apply(lambda x: [x])

    dataset = EvaluationDataset.from_list(df.to_dict("records"))

    # 2. Setup Backend
    os.makedirs("ragas_data", exist_ok=True)
    my_backend = LocalJSONLBackend("ragas_data")

    # 3. Config
    config = RunConfig(max_workers=1, timeout=400)

    # 4. Run
    print(
        "Starting evaluation... This will take a few minutes due to rate limit sleeps."
    )
    results = await agent_eval.arun(
        dataset=dataset, run_config=config, backend=my_backend, name="legal_audit_v1"
    )
    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)
    print(results.to_pandas()[["user_input", "response", "faithfulness"]])
    print("=" * 30)

    print("Done!")


if __name__ == "__main__":
    # CRITICAL FOR WINDOWS: This prevents the 'RLock' and 'Recursion' errors
    multiprocessing.freeze_support()
    asyncio.run(main())
