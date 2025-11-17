from pydantic import Field, BaseModel


class ModelParams(BaseModel):
    prompt: str = Field(description="Prompt. Passing as enumeration of key words")
    negative: str = Field(description="Negative prompt. Passing as enumeration of key words", default=" ")
    num_inference_steps: int = 30
    true_cfg_scale: int = 4
    guidance_scale: int = 1
