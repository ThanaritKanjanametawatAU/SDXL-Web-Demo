from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import load_model, generate_image
from io import BytesIO
from PIL import Image
import base64
import requests

app = FastAPI()

# Load the model when the app starts
model_path = "api/model/NovelAIv2-7.safetensors"
pipe = load_model(model_path)

# Define a request body using Pydantic
class ImageRequest(BaseModel):
    prompt: str
    height: int = 1216
    width: int = 832
    num_inference_steps: int = 28
    guidance_scale: float = 5.0
    clip_skip: int = 2
    seed: int = -1

# Helper function to encode image to Base64
def file_to_base64(image):
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return "data:image/png;base64," + base64_str

# Function to decode base64 or handle image URL
def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        # Fetch image from URL
        try:
            response = requests.get(encoding, timeout=30)
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image URL") from e

    # Handle Base64-encoded images
    if encoding.startswith("data:image/"):
        encoding = encoding.split(",")[1]

    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e

@app.post("/generate")
async def generate(request: ImageRequest):
    try:
        # Generate the image using the model
        image = generate_image(pipe, request.prompt, request.height, request.width, request.num_inference_steps, request.guidance_scale, request.clip_skip, request.seed)

        # Convert the image to base64 and return as Data URI
        img_base64 = file_to_base64(image)
        return {"image": img_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Stable Diffusion XL FastAPI is running."}

if __name__ == '__main__':
    import uvicorn
    # With Reload
    uvicorn.run(app, host="0.0.0.0", port=8000)