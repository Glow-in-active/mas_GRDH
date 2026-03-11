# Google Colab: генерация stego-датасета + загрузка в S3 (Yandex Object Storage)

Один файл, который можно копировать в Colab и запускать в разных сессиях.

## Что нужно заранее

- Бакет: `stegopractice`
- Папка (префикс): `Stego/`
- Доступы (не публикуйте их):
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION=ru-central1`
  - endpoint: `https://storage.yandexcloud.net`

## 0) Клонирование

```bash
git clone https://github.com/HXX5656/mas_GRDH.git
cd mas_GRDH
```

## 1) Зависимости

```bash
pip install -U gdown huggingface_hub transformers omegaconf pytorch-lightning einops scipy pillow tqdm torchvision opencv-python-headless boto3
```

## 2) Веса

### 2.1) Stable Diffusion v1.5 (ckpt)

```bash
mkdir -p weights
gdown "https://drive.google.com/uc?id=1ISeumzN-JrhyAacOPlh1IjAv6MKffogU" -O weights/v1-5-pruned.ckpt
```

### 2.2) CLIP text encoder (скачать целиком в папку)

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="weights/clip-vit-large-patch14",
    local_dir_use_symlinks=False,
)
print("OK: weights/clip-vit-large-patch14")
PY
```

## 3) S3 переменные (Yandex Object Storage)

В Colab можно просто выполнить (подставьте свои ключи):

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="ru-central1"
export S3_ENDPOINT_URL="https://storage.yandexcloud.net"
export S3_BUCKET="stegopractice"
export S3_PREFIX="Stego"
```

## 4) Генерация чанком `start/count` и загрузка в `Stego/`

Параметры:
- `dataset`: `chatgpt|laion|coco|flickr` (берётся из `text_prompt_dataset/*`)
- `start`: индекс строки с 0
- `count`: сколько строк обработать

Пример: **первые 1000 строк** `chatgpt`:
- `dataset=chatgpt`
- `start=0`
- `count=1000`

Запуск (можно менять `dataset/start/count` внизу):

```bash
python - <<'PY'
import os
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import torch
import torchvision.utils as vutils
from omegaconf import OmegaConf

REPO = Path(".").resolve()

DATASETS = {
    "laion": REPO / "text_prompt_dataset" / "laion_dataset.txt",
    "coco": REPO / "text_prompt_dataset" / "coco_dataset.txt",
    "flickr": REPO / "text_prompt_dataset" / "flickr_dataset.txt",
    "chatgpt": REPO / "text_prompt_dataset" / "chatgpt_dataset.txt",
}

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def s3_key(prefix: str, dataset: str, idx: int) -> str:
    p = prefix.strip("/")
    if p:
        return f"{p}/{dataset}/identity/{idx:08d}.png"
    return f"{dataset}/identity/{idx:08d}.png"

dataset = os.environ.get("DATASET", "chatgpt")
start = int(os.environ.get("START", "0"))
count = int(os.environ.get("COUNT", "1000"))

ckpt_path = REPO / "weights" / "v1-5-pruned.ckpt"
clip_dir = REPO / "weights" / "clip-vit-large-patch14"
config_path = REPO / "configs" / "stable-diffusion" / "ldm.yaml"

bucket = require_env("S3_BUCKET")
prefix = os.environ.get("S3_PREFIX", "Stego")
endpoint = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
region = os.environ.get("AWS_DEFAULT_REGION", "ru-central1")

s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region)

# --- load model ---
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import scripts.mapping_module as mapping_module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = OmegaConf.load(str(config_path))
try:
    config.model.params.cond_stage_config.params.version = str(clip_dir)
except Exception:
    pass

model = instantiate_from_config(config.model)
pl_sd = torch.load(str(ckpt_path), map_location=str(device), weights_only=False)
sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
model.load_state_dict(sd, strict=False)
model.to(device).eval()
sampler = DPMSolverSampler(model)

prompts_path = DATASETS[dataset]
lines = prompts_path.read_text(encoding="utf-8", errors="ignore").splitlines()
end = min(start + count, len(lines))
prompts = [p.strip() for p in lines[start:end] if p.strip()]

width = 512
height = 512
C = 4
f = 8
n_samples = 1

mapping = getattr(mapping_module, "ours_mapping")(bits=1)

uploaded = 0
skipped = 0

with torch.no_grad():
    for j, prompt in enumerate(prompts):
        idx = start + j
        key = s3_key(prefix, dataset, idx)
        if object_exists(s3, bucket=bucket, key=key):
            skipped += 1
            continue

        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([""])

        latent_shape = (n_samples, C, height // f, width // f)
        random_input = np.random.randint(0, 2, latent_shape)  # bits=1
        random_input_ori_sample: Optional[np.ndarray] = None
        if mapping.need_uniform_sampler:
            random_input_ori_sample = np.random.rand(*latent_shape)
        if mapping.need_gaussian_sampler:
            random_input_ori_sample = np.random.randn(*latent_shape)

        random_input_args = {
            "seed_kernel": np.random.randint(0, 2**32, 1),
            "seed_shuffle": np.random.randint(0, 2**32, 1),
        }

        init_latent = mapping.encode_secret(
            secret_message=random_input,
            ori_sample=random_input_ori_sample,
            **random_input_args,
        ).astype(np.float32)
        init_latent = torch.from_numpy(init_latent).to(device)

        shape = init_latent.shape[1:]
        z_0, _ = sampler.sample(
            steps=20,
            unconditional_conditioning=uc,
            conditioning=c,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=5.0,
            eta=0.0,
            order=2,
            x_T=init_latent,
            width=width,
            height=height,
            DPMencode=False,
            DPMdecode=True,
        )

        x0 = model.decode_first_stage(z_0)
        out_path = Path("/tmp") / f"{dataset}_{idx:08d}.png"
        vutils.save_image((x0 + 1) / 2, str(out_path))
        s3.upload_file(str(out_path), bucket, key)
        uploaded += 1

print(f"dataset={dataset} start={start} requested={count} actual={len(prompts)} uploaded={uploaded} skipped={skipped}")
PY
```

### Как запускать много сессий без пересечений

Берите фиксированный `count` (например 500) и для \(k\)-й сессии:

- `START = k * 500`
- `COUNT = 500`

В Colab можно задавать это переменными окружения перед запуском:

```bash
export DATASET=chatgpt
export START=0
export COUNT=500
```

