# Башкирэнерго RAG

## Описание

Проект для создания RAG (Retrieval-Augmented Generation) системы для обработки документов Башкирэнерго. Использует DeepSeek-R1 в качестве LLM и Nomic эмбеддинги для векторизации текста. Включает семантический чанкинг с использованием deepseek-r1 для более точного разбиения документов.

## Архитектура

- **DeepSeek-R1**: для генерации ответов и семантического чанкинга
- **Nomic-embed-text-v1.5**: для создания векторных эмбеддингов
- **Qdrant**: для хранения векторной базы данных
- **Langchain**: для построения RAG-конвейера
- **OCR**: для обработки сканированных документов

## Запуск

### Подготовка

1. Убедитесь, что у вас установлены:
   - Docker и Docker Compose
   - NVIDIA Docker runtime (для GPU-ускорения)
   - Ollama с моделью `deepseek-r1:8b`

2. Установите модель Ollama:
   ```bash
   ollama pull deepseek-r1:8b
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
docker run -it --rm --gpus all `
  -v ${PWD}/documents:/app/documents `
  -v ${PWD}/output:/app/output `
  -v ${PWD}/scripts:/app/scripts `
  my-pipeline bash

```bash
# Сборка образа
docker build -t bashkir-rag .

# Запуск обработки документов
docker run --gpus all -v ./documents:/app/documents -v ./output:/app/output -v ./.env:/app/.env -e HUGGINGFACE_HUB_TOKEN --network="host" bashkir-rag
```
#запуск скрипта в контейнере 
uv run python scripts/ai_chunking.py
## Структура проекта

- `scripts/parse_docs_ocr.py`: конвертация PDF в Markdown с OCR
- `scripts/semantic_chunking.py`: семантическое разбиение документов с помощью deepseek-r1
- `scripts/ai_chunking.py`: создание векторной базы в Qdrant из семантических чанков
- `scripts/generat_answer.py`: тестирование подключения к Ollama
- `pyproject.toml`: зависимости проекта
- `Dockerfile`: определение Docker-образа
- `docker-compose.yml`: конфигурация для запуска всех сервисов