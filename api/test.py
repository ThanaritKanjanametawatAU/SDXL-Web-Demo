import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from safetensors.torch import load_file

torch.manual_seed(315514780744)

# Define the path to the NovelAIv2-7 safetensors model
model_path = "model/NovelAIv2-7.safetensors"

# Load the StableDiffusionXLPipeline, specify any pre-configured model that the safetensors file is based on
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

# Load safetensors into the specific components of the pipeline
state_dict = load_file(model_path)

# Print the keys in the state_dict to identify the components
if any(key.startswith("unet") for key in state_dict.keys()):
    pipe.unet.load_state_dict(state_dict, strict=False)
    print("Loaded state_dict into the UNet component.")

if any(key.startswith("diffusion") for key in state_dict.keys()):
    pipe.diffusion.load_state_dict(state_dict, strict=False)
    print("Loaded state_dict into the pipeline components.")




# Set up the DDIMScheduler as the sampling method (alternative to DPM++)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Move the pipeline to the GPU
pipe = pipe.to("cuda")

# Example prompt for generating an image
prompt = "1boy, pale skin, {purple eyes}, sanpaku, very long hair, pink hair, teeth, bloomers, slippers, waist apron, handgun, lollipop, spoon, best quality, amazing quality, very aesthetic, absurdres, masterpiece"
negative_prompt = "nsfw, nudity, lowres, bad, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], text"

# Generate the image
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=1216, width=832, num_inference_steps=28, guidance_scale=5, clip_skip=2).images[0]

# Save the generated image
image.save("output_image.png")
