from transformers import pipeline
from PIL import Image, ImageFile
from openai import AsyncOpenAI
import os
import logging
import sys 
import pandas as pd 

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
        result = response.results[0]  # First moderation result

        print(response)
        # Log the input and response
        logging.info(f"Input: {message}")
        logging.info(f"Moderation Results: {response}")
        print("0")
        flagged_categories = {cat: getattr(result.categories, cat) for cat in vars(result.categories)}
        flagged_scores = {f"S{cat}": getattr(result.category_scores, cat) for cat in vars(result.category_scores)}
        print("1")

        # Prepare log entry as DataFrame
        log_entry = pd.DataFrame([{ "Input": message, **flagged_categories, **flagged_scores }])

        print("2")

        # Log the result in a CSV file
        try:
            df = pd.read_csv(LOG_FILE)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Input"] + list(flagged_categories.keys()) + list(flagged_scores.keys()))

        # Concatenate new log entry
        if df.empty:
            df = log_entry  # Directly assign log_entry if df is empty
        else:
            df = pd.concat([df, log_entry], ignore_index=True)

        # Save back to CSV
        df.to_csv(LOG_FILE, index=False)


        if response.results[0].flagged:
            return False
        return True
    except Exception as e:
        print(f"Error: {e}")
        return await message_is_safe(message, apikey)
