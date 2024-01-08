from dotenv import load_dotenv
import os

load_dotenv()

secrets = {
    "hf_token": os.getenv("hf_token"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_ORGANIZATION": os.getenv("OPENAI_ORGANIZATION"),
}
