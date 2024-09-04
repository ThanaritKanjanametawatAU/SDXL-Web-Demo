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

# API endpoint URL
url = "http://127.0.0.1:8000/generate"

# Define the request payload
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
        # Display the image using matplotlib
        # plt.imshow(img)
        # plt.axis('off')  # Turn off axis
        # plt.show()

        # Save the image to a file
        img.save("1.png")
    else:
        print("Failed to decode the image")

else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print("Error:", response.text)
