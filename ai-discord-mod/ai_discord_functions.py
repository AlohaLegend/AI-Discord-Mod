# from transformers import pipeline
# from PIL import Image, ImageFile
from openai import AsyncOpenAI
import os
import logging
import json
import csv
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    filename="moderation_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

LOG_FILE = "moderation_log.csv"

load_dotenv()

# Load global moderation thresholds from servers.json
try:
    with open("servers.json", "r") as f:
        servers = json.load(f)
        moderation_thresholds = servers.get("moderation_thresholds", {})
except:
    moderation_thresholds = {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Image moderation temporarily disabled
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# async def image_is_safe(sensitivity):
#     from transformers import pipeline
#     vqa_pipeline = pipeline("visual-question-answering")
    
#     image = Image.open("toModerate.jpeg")
#     question = "Does the image contain pornographic, adult, gore, sexual, or other NSFW content?"
#     sensitivity = 1 - sensitivity
#     result = vqa_pipeline(image, question, top_k=1)[0]
#     answer = result["answer"].lower()

#     print(result)

#     if result["score"] > sensitivity and answer.startswith("y"):
#         return False
#     elif result["score"] < sensitivity and answer.startswith("n"):
#         return False
#     return True

async def message_is_safe(message, apikey, servers, guild_id):
    try:
        response = await aclient.moderations.create(input=message)
        result = response.results[0]

        logging.info(f"Input: {message}")
        logging.info(f"Moderation Results: {response}")

        flagged_categories = {cat: getattr(result.categories, cat) for cat in vars(result.categories)}
        flagged_scores = {f"S{cat}": getattr(result.category_scores, cat) for cat in vars(result.category_scores)}

        log_row = [message] + list(flagged_categories.values()) + list(flagged_scores.values())
        header = ["Input"] + list(flagged_categories.keys()) + list(flagged_scores.keys())
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(log_row)

        thresholds = servers.get(str(guild_id), {}).get("moderation_thresholds", {})

        for cat, is_flagged in vars(result.categories).items():
            base_cat = cat.split("/")[0]
            threshold = thresholds.get(base_cat, 0.8)
            score = getattr(result.category_scores, cat, 0.0)

            if is_flagged and score >= threshold:
                return False, base_cat, score, threshold

        return True, None, None, None

    except Exception as e:
        print(f"Error: {e}")
        return False, None, None, None
