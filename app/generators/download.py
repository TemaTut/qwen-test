from huggingface_hub import snapshot_download

model_repo = "AI-Porn/pornworks-nude-people-photo-realistic-nsfw-flux-1d-checkpoint"

local_dir = "../../models"

snapshot_download(repo_id=model_repo, local_dir=local_dir)
print(f"Модель скачана в {local_dir}")