from pydantic_ai.models.groq import GroqModel
from dotenv import load_dotenv
import os
from pydantic_ai.providers.groq import GroqProvider


load_dotenv()


def get_model():
    llm = os.getenv("MODEL_CHOICE")
    api_key = os.getenv("GROQ_API_KEY")

    return GroqModel(
        model_name=llm,
        provider=GroqProvider(api_key=api_key)
    )
