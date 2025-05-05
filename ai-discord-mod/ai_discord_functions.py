from transformers import pipeline
from PIL import Image, ImageFile
from openai import AsyncOpenAI
import os
import logging
import sys 
import pandas as pd 
import json

from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    filename="moderation_log.txt",  # Log file name
    level=logging.INFO,             # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)
# Define log file
LOG_FILE = "moderation_log.csv"


load_dotenv()

# Load global moderation thresholds from servers.json
try:
    with open("servers.json", "r") as f:
        servers = json.load(f)
        moderation_thresholds = servers.get("moderation_thresholds", {})
except:
    moderation_thresholds = {}  # Fallback to empty


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

ImageFile.LOAD_TRUNCATED_IMAGES = True

vqa_pipeline = pipeline("visual-question-answering")

async def image_is_safe(sensitivity):
    image =  Image.open("toModerate.jpeg")
    question = "Does the image contain pornographic, adult, gore, sexual, or other NSFW content?"
    sensitivity = 1 - sensitivity
    result = vqa_pipeline(image, question, top_k=1)[0]
    answer = result["answer"].lower()

    print(result)

    if result["score"] > sensitivity and answer.startswith("y"):
        return False
    elif result["score"] < sensitivity and answer.startswith("n"):
        return False
    return True


async def message_is_safe(message, apikey):
    try:
        response = await aclient.moderations.create(input=message)
        result = response.results[0]

        logging.info(f"Input: {message}")
        logging.info(f"Moderation Results: {response}")

        flagged_categories = {cat: getattr(result.categories, cat) for cat in vars(result.categories)}
        flagged_scores = {f"S{cat}": getattr(result.category_scores, cat) for cat in vars(result.category_scores)}

        log_entry = pd.DataFrame([{ "Input": message, **flagged_categories, **flagged_scores }])

        try:
            df = pd.read_csv(LOG_FILE)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Input"] + list(flagged_categories.keys()) + list(flagged_scores.keys()))

        if df.empty:
            df = log_entry
        else:
            df = pd.concat([df, log_entry], ignore_index=True)

        df.to_csv(LOG_FILE, index=False)

        for cat, is_flagged in vars(result.categories).items():
            base_cat = cat.split("/")[0]
            threshold = moderation_thresholds.get(base_cat, 0.8)
            score = getattr(result.category_scores, cat, 0.0)

            if is_flagged and score >= threshold:
                return False, base_cat, score, threshold  # ✅ modified return

        return True, None, None, None  # ✅ modified return

    except Exception as e:
        print(f"Error: {e}")
        return False, None, None, None  # ✅ modified fallback return
