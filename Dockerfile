FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 1. Системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Копируем pyproject.toml
COPY pyproject.toml .

# 3. Устанавливаем uv и зависимости в одном RUN
RUN pip install uv && \
    uv sync --no-cache \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

# 4. Копируем весь код
COPY . .

# 5. Проверка GPU
RUN python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

CMD ["uv", "run", "scripts/ai_chunking.py"]