from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch


pipe = StableDiffusionXLPipeline.from_single_file("./models/anime-2d-ill_v3.fp16.safetensors", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "best quality,masterpiece,absurdres,newest,(narrative-rich composition), (dramatic angle,dynamic angle), (kim hana:1.15),(daeho cha:0.75),(kyu yong eom,rabbit \(tukenitian\):0.95),(quuni,housou-kun:0.85),(gyhgjart:0.95),(ayas style:1.2),(dhr:1.15),(nyatabe:0.75), (3d:1.25), 1girl, solo, long hair, breasts, looking at viewer, large breasts, long sleeves, red eyes, dress, closed mouth, sitting, blue hair, purple hair, pantyhose, horns, ch'en (arknights)"
prompt_2 = "black dress, feet, black pantyhose, toes, soles, no shoes, dragon horns, reflection, legs up, dragon girl, foot focus, feet up"
negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (watermark, bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++",
)

image = pipe(
    prompt=prompt,
    prompt_2=prompt_2,
    negative_promt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4,
).images[0]

image.save("anime_18_3.png")
