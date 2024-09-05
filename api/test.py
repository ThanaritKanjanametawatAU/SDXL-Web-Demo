import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
import k_diffusion.sampling



# samplers_k_diffusion = [
#     ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {'scheduler': 'karras'}),
#     ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
#     ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde'], {'scheduler': 'exponential', "brownian_noise": True}),
#     ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
#     ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
#     ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
#     ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
#     ('Euler', 'sample_euler', ['k_euler'], {}),
#     ('LMS', 'sample_lms', ['k_lms'], {}),
#     ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
#     ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "second_order": True}),
#     ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
#     ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
#     ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
# ]
# 
# schedulers = [
#     Scheduler('automatic', 'Automatic', None),
#     Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
#     Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
#     Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
#     Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
#     Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
#     Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
#     Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
#     Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
#     Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
#     Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
#     Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
# ]



# Define the path to the NovelAIv2-7 safetensors model
model_path = "model/NovelAIv2-7.safetensors"

# Load the StableDiffusionXLPipeline, specify any pre-configured model that the safetensors file is based on
pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)

# Load safetensors into the specific components of the pipeline


# Set up the DDIMScheduler as the sampling method (alternative to DPM++)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# Move the pipeline to the GPU
pipe = pipe.to("cuda")

# Example prompt for generating an image
prompt = "{1boy, lavender background, full body, purple eyes, short hair, side ponytail, aqua hair, toned, witch hat, raglan sleeves, black uwabaki, spiked bracelet, year 2022, best quality, amazing quality, very aesthetic, absurdres"
negative_prompt = "nsfw, nudity, lowres, bad, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract], text"

# Generate the image
torch.manual_seed(315514780744)
image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=1216, width=832, num_inference_steps=28, guidance_scale=5, clip_skip=2).images[0]

# Save the generated image
image.save("1.png")
