FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем pyproject.toml и uv.lock
COPY pyproject.toml uv.lock ./

# Устанавливаем uv и зависимости (с кэшем)
RUN pip install uv && \
    UV_HTTP_TIMEOUT=60 uv sync

# Копируем весь код
COPY . .

# Проверка GPU
RUN python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

CMD ["uv", "run", "scripts/run_full_pipeline.py"]