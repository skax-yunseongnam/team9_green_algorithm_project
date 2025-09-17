
import os
from dotenv import load_dotenv

def load_settings():
    load_dotenv()
    return {
        "AOAI_API_KEY": os.getenv("AOAI_API_KEY"),
        "AOAI_ENDPOINT": os.getenv("AOAI_ENDPOINT"),
        "AOAI_DEPLOY_DEFAULT": os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini"),
    }
