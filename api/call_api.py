import os
from os.path import exists

import requests
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Helper function to decode base64 or handle image URL
def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        try:
            response = requests.get(encoding, timeout=30)
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"Error fetching image from URL: {e}")
            return None

    # Handle Base64-encoded images
    if encoding.startswith("data:image/"):
        encoding = encoding.split(",")[1]

    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        print(f"Error decoding Base64 image: {e}")
        return None



def generate_image(url="http://127.0.0.1:8000/generate", payload=None, save_path="data/gens/something/1.png"):
    # Define the payload for the POST
    if not payload:
        payload = {
            "prompt": "1boy, pale skin, {purple eyes}, sanpaku, very long hair, pink hair, teeth, bloomers, slippers, waist apron, handgun, lollipop, spoon, best quality, amazing quality, very aesthetic, absurdres, masterpiece",
            "height": 1216,
            "width": 832,
            "num_inference_steps": 28,
            "guidance_scale": 5.0,
            "clip_skip": 2
        }



    # Send POST request to the API
    response = requests.post(url, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        print("Image generated successfully!")

        # Get the base64 encoded image from the response
        img_data_uri = response.json()["image"]

        # Decode the Base64-encoded image to an image object
        img = decode_base64_to_image(img_data_uri)

        if img:
            dir = os.path.dirname(save_path)
            if not exists(dir):
                os.makedirs(dir)
            img.save(save_path)
        else:
            print("Failed to decode the image")

    else:
        print(f"Failed to generate image. Status code: {response.status_code}")
        print("Error:", response.text)
