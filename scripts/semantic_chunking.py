"""
Семантический чанкинг документов с использованием deepseek-r1
"""
import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Инициализация LLM
llm = OllamaLLM(
    model="deepseek-r1:8b",
    base_url="http://host.docker.internal:11434",
    temperature=0.1
)

# Шаблон промпта для семантического чанкинга
semantic_chunking_prompt = PromptTemplate(
    input_variables=["text", "chunk_size"],
    template="""
    Ты — эксперт по анализу юридических и деловых документов. Раздели следующий текст на логически связанные семантические блоки (чанки).
    Каждый чанк должен содержать полную мысль или связанную группу мыслей, не превышающую {chunk_size} токенов.
    ВАЖНО: 
    - Сохраняй контекст и логическую целостность каждого чанка
    - Не разбивай текст посередине предложений
    - Учитывай структуру документа (разделы, подразделы, статьи)
    - Предпочтительно разбивай на чанки в местах естественных пауз
    
    Текст для разбиения:
    {text}
    
    Верни результат в формате JSON массива строк, где каждая строка — это отдельный семантический чанк:
    [ "чанк1", "чанк2", ... ]
    """
)

def semantic_chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """
    Семантически разбивает текст на чанки с помощью deepseek-r1
    """
    # Если текст короче размера чанка, возвращаем как есть
    if len(text) < chunk_size * 3:  # грубая оценка
        return [text]
    
    # Вызов LLM для семантического разбиения
    chain = semantic_chunking_prompt | llm
    response = chain.invoke({"text": text, "chunk_size": chunk_size})
    
    try:
        # Пытаемся распарсить JSON из ответа
        # Убираем возможные маркеры кода
        clean_response = re.sub(r'```json\s*|\s*```', '', response, flags=re.DOTALL)
        chunks = json.loads(clean_response.strip())
        
        if isinstance(chunks, list):
            return [chunk for chunk in chunks if chunk.strip()]
        else:
            # Если ответ не в формате массива, возвращаем как один чанк
            return [text]
    except json.JSONDecodeError:
        # Если не удалось распарсить JSON, возвращаем простое разбиение
        print("⚠️ Не удалось распарсить JSON, используем простое разбиение")
        return simple_chunk_text(text, chunk_size)

def simple_chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """
    Простое разбиение текста на чанки по количеству слов
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word)
        
        if current_length > chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def process_markdown_files(input_dir: Path, output_dir: Path, chunk_size: int = 512):
    """
    Обрабатывает все markdown файлы в директории и создает семантически чанкованные версии
    """
    output_dir.mkdir(exist_ok=True)
    
    md_files = list(input_dir.glob("*.md"))
    print(f"Найдено {len(md_files)} markdown файлов для обработки")
    
    for md_file in md_files:
        print(f"Обработка: {md_file.name}")
        
        # Читаем содержимое файла
        content = md_file.read_text(encoding="utf-8")
        
        # Разделяем YAML заголовок и основной текст
        yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
        
        if yaml_match:
            yaml_header = yaml_match.group(1)
            text_content = yaml_match.group(2)
            metadata = yaml.safe_load(yaml_header)
        else:
            text_content = content
            metadata = {}
        
        # Разбиваем текст на семантические чанки
        chunks = semantic_chunk_text(text_content, chunk_size)
        
        print(f"  Создано {len(chunks)} чанков")
        
        # Сохраняем каждый чанк как отдельный файл
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{md_file.stem}_chunk_{i:03d}.md"
            chunk_path = output_dir / chunk_filename
            
            # Обновляем метаданные для чанка
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["original_file"] = md_file.name
            chunk_metadata["chunk_count"] = len(chunks)
            
            # Формируем YAML заголовок для чанка
            yaml_header = yaml.dump(chunk_metadata, default_flow_style=False, allow_unicode=True)
            chunk_content = f"---\n{yaml_header}---\n\n{chunk}"
            
            chunk_path.write_text(chunk_content, encoding="utf-8")
            print(f"  Сохранен чанк: {chunk_filename}")

def main():
    input_dir = Path("/app/output")  # Директория с исходными markdown файлами
    output_dir = Path("/app/semantic_chunks")  # Директория для семантических чанков
    chunk_size = 512  # Приблизительный размер чанка в токенах
    
    print("🚀 Запуск семантического чанкинга")
    print(f"  Входная директория: {input_dir}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  Размер чанка: {chunk_size}")
    
    process_markdown_files(input_dir, output_dir, chunk_size)
    
    print("\n✅ Семантический чанкинг завершен")

if __name__ == "__main__":
    main()