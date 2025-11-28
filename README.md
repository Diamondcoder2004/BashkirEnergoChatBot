# Башкирэнерго RAG

## Описание

Проект для создания RAG (Retrieval-Augmented Generation) системы для обработки документов Башкирэнерго. Использует Saiga-llama3 в качестве LLM, MiniLM-L12-v2 эмбеддинги для векторизации текста и Qwen3-Reranker-0.6B для переупорядочивания документов. Включает семантический чанкинг и контекстное сжатие для более точного разбиения и обработки документов.

## Архитектура

- **Saiga-llama3**: для генерации ответов и контекстного сжатия
- **MiniLM-L12-v2**: для создания векторных эмбеддингов
- **Qwen3-Reranker-0.6B**: для переупорядочивания документов по релевантности
- **Qdrant**: для хранения векторной базы данных
- **Langchain**: для построения RAG-конвейера
- **OCR**: для обработки сканированных документов

## Запуск

### Подготовка

1. Убедитесь, что у вас установлены:
   - Docker и Docker Compose
   - NVIDIA Docker runtime (для GPU-ускорения)
   - Ollama с моделями `bambucha/saiga-llama3` и `dengcao/Qwen3-Reranker-0.6B:latest`

2. Установите модели Ollama:
   ```bash
   ollama pull bambucha/saiga-llama3
   ollama pull dengcao/Qwen3-Reranker-0.6B:latest
   ```

3. Создайте файл `.env` с токеном Hugging Face:
   ```env
   HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
   ```

4. Положите PDF-документы в папку `documents/`

### Запуск с помощью Docker Compose

```bash
docker-compose up --build
```

### Альтернативный запуск
```bash
docker run -it --rm --gpus all \
  -v ${PWD}/documents:/app/documents \
  -v ${PWD}/output:/app/output \
  -v ${PWD}/scripts:/app/scripts \
  my-pipeline bash

# Сборка образа
docker build -t bashkir-rag .

# Запуск обработки документов
docker run --gpus all -v ./documents:/app/documents -v ./output:/app/output -v ./.env:/app/.env -e HUGGINGFACE_HUB_TOKEN --network="host" bashkir-rag

# Запуск скрипта в контейнере 
uv run python scripts/ai_chunking.py
```

## Структура проекта

- `scripts/parse_docs_ocr.py`: конвертация PDF в Markdown с OCR
- `scripts/semantic_chunking.py`: семантическое разбиение документов с помощью HuggingFace эмбеддингов
- `scripts/ai_chunking.py`: создание векторной базы в Qdrant из семантических чанков
- `scripts/generat_answer.py`: тестирование подключения к Ollama
- `pyproject.toml`: зависимости проекта
- `Dockerfile`: определение Docker-образа
- `docker-compose.yml`: конфигурация для запуска всех сервисов
- `reranker.py`: модуль для переупорядочивания документов