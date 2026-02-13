from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from dotenv import load_dotenv
import os

load_dotenv()

model = GeminiModel(model="gemini-2.5-flash", api_key=os.environ.get("GOOGLE_API_KEY"))
task_completion_metric = AnswerRelevancyMetric(model=model)
