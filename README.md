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

# Сборка образа из Dockerfile
docker build -t bashkir-rag .

# Актуальный способ сборки и запуска через docker-compose 
docker-compose up -d document-processor

# Подключаемся к контейнеру
docker exec -it bashkir-rag-processor bash

# Запуск скрипта в контейнере

uv run python scripts/parse_docs_ocr.py
uv run python scripts/semantic_chunking.py  
uv run python scripts/chunk_embedding_qdrant.py
uv run python scripts/run_full_pipeline.py  #  интерактивное меню
```
# посчитать количество файлов
find /app/data/documents -type f | wc -l


# Для rag-api
Для изменений в коде (.py файлы):
bash
# Просто пересобери (кэш зависимостей сохранится)
docker-compose up -d --build rag-api

# Для изменений в .env:

bash
# Перезапусти без сборки (env подхватится автоматически)
docker-compose down
docker-compose up -d rag-api

## Структура проекта

- `scripts/parse_docs_ocr.py`: конвертация PDF в Markdown с OCR
- `scripts/semantic_chunking.py`: семантическое разбиение документов с помощью HuggingFace эмбеддингов
- `scripts/ai_chunking.py`: создание векторной базы в Qdrant из семантических чанков
- `scripts/generat_answer.py`: тестирование подключения к Ollama
- `pyproject.toml`: зависимости проекта
- `Dockerfile`: определение Docker-образа
- `docker-compose.yml`: конфигурация для запуска всех сервисов
- `reranker.py`: модуль для переупорядочивания документов