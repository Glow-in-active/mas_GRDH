import gc
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import json
import numpy as np
import torch
import torchvision.utils as vutils
from tqdm.auto import tqdm
from botocore.config import Config
from botocore.exceptions import ClientError
from omegaconf import OmegaConf

# ====== ЧАНК (меняй между сессиями) ======
DATASET = "own5"   # chatgpt|laion|coco|flickr
START = 0             # 0-index
COUNT = 30000            # например 500 или 1000

# ====== S3 (Yandex Object Storage) ======
# Вариант A (рекомендую): Colab Secrets
# from google.colab import userdata
# AWS_ACCESS_KEY_ID = userdata.get("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = userdata.get("AWS_SECRET_ACCESS_KEY")
#
# Вариант B: вставить ключи прямо сюда (не сохраняй/не публикуй ноутбук с ними)
AWS_ACCESS_KEY_ID = "YCAJEzMeDE5s_87MnO5vpZpJa"
AWS_SECRET_ACCESS_KEY = "YCOJn7_m0hoSnLOA6BrnwmJ-PtUYyYXZ_PnlNDnM"

AWS_DEFAULT_REGION = "ru-central1"
S3_ENDPOINT_URL = "https://storage.yandexcloud.net"
S3_BUCKET = "stegopractice"
S3_PREFIX = "Stego"

def _strip_quotes(x: str) -> str:
    return x.strip().strip('"').strip("'")

AWS_DEFAULT_REGION = _strip_quotes(AWS_DEFAULT_REGION)
S3_ENDPOINT_URL = _strip_quotes(S3_ENDPOINT_URL)
S3_BUCKET = _strip_quotes(S3_BUCKET)
S3_PREFIX = _strip_quotes(S3_PREFIX)

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

REPO = Path(".").resolve()

DATASETS = {
    "laion": REPO / "text_prompt_dataset" / "laion_dataset.txt",
    "coco": REPO / "text_prompt_dataset" / "coco_dataset.txt",
    "flickr": REPO / "text_prompt_dataset" / "flickr_dataset.txt",
    "chatgpt": REPO / "text_prompt_dataset" / "chatgpt_dataset.txt",
    "own1": REPO / "text_prompt_dataset" / "own1_dataset.txt",
    "own2": REPO / "text_prompt_dataset" / "own2_dataset.txt",
    "own3": REPO / "text_prompt_dataset" / "own1_dataset.txt",
    "own4": REPO / "text_prompt_dataset" / "own2_dataset.txt",
    "own5": REPO / "text_prompt_dataset" / "own1_dataset.txt",
}

def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "")
        status = (e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if status == 404 or code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def s3_key(prefix: str, dataset: str, idx: int) -> str:
    p = prefix.strip("/")
    if p:
        return f"{p}/{dataset}/identity/{idx:08d}.png"
    return f"{dataset}/identity/{idx:08d}.png"

# ВАЖНО для Yandex: path-style + s3v4
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    region_name=AWS_DEFAULT_REGION,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

# --- load model ---
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import scripts.mapping_module as mapping_module

ckpt_path = REPO / "weights" / "v1-5-pruned-emaonly.fp16.ckpt"
clip_dir = REPO / "weights" / "clip-vit-large-patch14"
config_path = REPO / "configs" / "stable-diffusion" / "ldm.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Если до этого уже падало по памяти — очистим кеш.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

config = OmegaConf.load(str(config_path))
try:
    config.model.params.cond_stage_config.params.version = str(clip_dir)
except Exception:
    pass

model = instantiate_from_config(config.model)

# КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: грузим чекпойнт на CPU, иначе можно словить CUDA OOM.
pl_sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
model.load_state_dict(sd, strict=False)

# Ещё одно ключевое: переводим в fp16 до переноса на GPU (экономия памяти).
model = model.half().to(device).eval()
sampler = DPMSolverSampler(model)

prompts_path = DATASETS[DATASET]
lines = prompts_path.read_text(encoding="utf-8", errors="ignore").splitlines()
end = min(START + COUNT, len(lines))
prompts = [p.strip() for p in lines[START:end] if p.strip()]

width = 512
height = 512
C = 4
f = 8
n_samples = 1

mapping = getattr(mapping_module, "ours_mapping")(bits=1)

uploaded = 0
skipped = 0

with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
    t0 = time.time()
    pbar = tqdm(prompts, desc=f"gen+upload {DATASET} [{START}:{START+len(prompts)}]", unit="img")
    for j, prompt in enumerate(pbar):
        idx = START + j
        key = s3_key(S3_PREFIX, DATASET, idx)
        if object_exists(s3, bucket=S3_BUCKET, key=key):
            skipped += 1
            pbar.set_postfix(uploaded=uploaded, skipped=skipped)
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

        seed_kernel = int(np.random.randint(0, 2**32))
        seed_shuffle = int(np.random.randint(0, 2**32))
        random_input_args = {"seed_kernel": seed_kernel, "seed_shuffle": seed_shuffle}

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
        out_path = Path("/tmp") / f"{DATASET}_{idx:08d}.png"
        vutils.save_image((x0 + 1) / 2, str(out_path))
        s3.upload_file(str(out_path), S3_BUCKET, key)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key[:-4] + ".json",
            Body=json.dumps(
                {
                    "dataset": DATASET,
                    "idx": idx,
                    "prompt": prompt,
                    "seed_kernel": seed_kernel,
                    "seed_shuffle": seed_shuffle,
                    "mapping_func": "ours_mapping",
                    "bit_num": 1,
                    "dpm_steps": 20,
                    "dpm_order": 2,
                    "scale": 5.0,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )
        uploaded += 1
        pbar.set_postfix(uploaded=uploaded, skipped=skipped)

dt = time.time() - t0
print(f"dataset={DATASET} start={START} requested={COUNT} actual={len(prompts)} uploaded={uploaded} skipped={skipped} elapsed_s={dt:.1f} rate_img_s={(uploaded/max(dt,1e-6)):.3f}")


