import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, \
    EulerDiscreteScheduler
# ######################## diffusion models ######################
diffusion_model = "runwayml/stable-diffusion-v1-5"

num_steps = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler_list = ["DPMSolverMultistep", "EulerAncestralDiscrete", "EulerDiscrete"]

# if device == "cpu":
#     pipeline = DiffusionPipeline.from_pretrained(diffusion_model)
# else:
#     pipeline = DiffusionPipeline.from_pretrained(diffusion_model, torch_dtype=torch.float16)

pipeline = DiffusionPipeline.from_pretrained(diffusion_model)
pipline = pipeline.to(device)

def generate_cards(prompt, num_images=4, scheduler="EulerDiscrete", num_steps=20):
    # generate images, return a list of (image,seed)
    # e.g [(im1, 1), (im2, 2), (im3, 3), (im4, 4)]
    images = []
    if scheduler == "EulerAncestralDiscrete":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DPMSolverMultistep":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "EulerDiscrete":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    # generate
    for i in range(num_images):
        seed = random.randint(0, 4294967295)
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipeline(prompt, generator=generator, num_inference_steps=num_steps).images[0]
        images.append((image, seed))

    return images


def modify(prompt, input_image, seed, num_steps=20):
    # generate 3 images with the same seed using different schedulers
    # return 4 images, the first image is the current image
    images = [(input_image, seed)]
    for scheduler in scheduler_list:
        if scheduler == "EulerAncestralDiscrete":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "DPMSolverMultistep":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        elif scheduler == "EulerDiscrete":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipeline(prompt, generator=generator, num_inference_steps=num_steps).images[0]
        images.append((image, seed))

    return images