from transformers import pipeline
from PIL import Image, ImageFile
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


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
        print(response)
        if response.results[0].flagged:
            return False
        return True
    except Exception as e:
        print(f"Error: {e}")
        return await message_is_safe(message, apikey)
