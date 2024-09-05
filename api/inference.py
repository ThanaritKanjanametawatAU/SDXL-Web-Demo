import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from safetensors.torch import load_file
import torch


# Load the StableDiffusionXLPipeline and your NovelAI model
def load_model(model_path: str):
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     torch_dtype=torch.float16)

    # Load custom model weights
    state_dict = load_file(model_path)
    pipe.unet.load_state_dict(state_dict, strict=False)

    # Set custom scheduler (e.g., DDIM)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Move pipeline to GPU
    pipe = pipe.to("cuda")
    return pipe


def generate_image(pipe, prompt: str, height: int = 1216, width: int = 832, num_inference_steps: int = 28,
                   guidance_scale: float = 5, clip_skip: int = 2, seed: int = -1):
    # Generate image using the provided parameters
    if seed != -1:
        torch.manual_seed(seed)
    else:
        # Random seed for each generation
        seed_used = torch.randint(0, 2 ** 32, (1,)).item()
        torch.manual_seed(seed_used)

    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale, clip_skip=clip_skip).images[0]
    return image
