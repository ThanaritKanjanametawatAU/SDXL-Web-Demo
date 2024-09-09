import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import k_diffusion.sampling
from safetensors.torch import load_file
import torch


# Load the StableDiffusionXLPipeline and your NovelAI model
def load_model(model_path: str):
    pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)


    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move pipeline to GPU
    pipe = pipe.to("cuda")
    return pipe


def generate_image_backend(pipe, prompt: str, height: int = 1216, width: int = 832, num_inference_steps: int = 28,
                   guidance_scale: float = 5, clip_skip: int = 2, seed: int = -1):
    # Generate image using the provided parameters
    seed_used = seed
    if seed == -1:
        # Random seed for each generation
        seed_used = torch.randint(0, 2 ** 32, (1,)).item()

    torch.manual_seed(seed_used)

    image = pipe(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale, clip_skip=clip_skip).images[0]
    return image
