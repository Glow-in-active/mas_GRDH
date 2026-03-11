# Как развернуть и запустить `mas_GRDH`

Ниже — минимальные шаги для запуска генерации/оценки в этом форке Stable Diffusion.

## Структура весов (рекомендуется)

- `weights/v1-5-pruned.ckpt` — Stable Diffusion v1.5 (чекпойнт `.ckpt`)
- `weights/clip-vit-large-patch14/` — CLIP text encoder (папка модели целиком)

В конфиге уже прописано, что CLIP берётся из `../weights/clip-vit-large-patch14`:
- `configs/stable-diffusion/ldm.yaml` → `model.cond_stage_config.params.version`

## Вариант A: Google Colab (команды из ноутбука)

```bash
git clone https://github.com/HXX5656/mas_GRDH.git
cd mas_GRDH
```

### 1) Установить зависимости

```bash
pip install -U gdown huggingface_hub pytorch-lightning omegaconf transformers einops opencv-python scipy matplotlib pillow tqdm torchvision
```

### 2) Скачать веса Stable Diffusion v1.5

```bash
gdown "https://drive.google.com/uc?id=1ISeumzN-JrhyAacOPlh1IjAv6MKffogU" -O weights/v1-5-pruned.ckpt
```

### 3) Скачать CLIP в локальную папку `weights/`

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="weights/clip-vit-large-patch14",
    local_dir_use_symlinks=False,
)
print("Downloaded CLIP to weights/clip-vit-large-patch14")
PY
```

### 4) Запустить тест

```bash
cd scripts
python txt2img.py \
  --ckpt ../weights/v1-5-pruned.ckpt \
  --config ../configs/stable-diffusion/ldm.yaml \
  --dpm_steps 20 \
  --dpm_order 2 \
  --scale 5. \
  --test_prompts ./test_prompts.txt \
  --attack_layer identity \
  --mapping_func ours_mapping
```

Результаты будут в `scripts/outputs/`.

## Вариант B: Локально (Linux)

1) Убедитесь, что установлен PyTorch под вашу CUDA/CPU (по инструкции с сайта PyTorch).
2) Далее шаги те же, что и для Colab: зависимости → `weights/` → запуск из `scripts/`.

## Примечания

- `--test_prompts` — это **путь к файлу с промптами** (по одному промпту на строку).
- Чекпойнт SD задаётся флагом `--ckpt`, поэтому его можно хранить где угодно, но удобнее в `weights/`.

