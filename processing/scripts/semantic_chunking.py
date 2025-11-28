"""
Семантический чанкинг документов с использованием OllamaEmbeddings и SemanticChunker
"""
import os
import re
import yaml
from pathlib import Path
from langchain_ollama import OllamaEmbeddings  # ИСПРАВЬ ИМПОРТ!
from langchain_experimental.text_splitter import SemanticChunker
from typing import List

# Инициализация эмбеддингов для чанкинга
ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
chunker_embeddings = OllamaEmbeddings(
    model="ognivo777/rubert-mini-frida:latest",  # ИСПРАВЬ: model вместо model_name
    base_url=ollama_host
)
# Остальной код без изменений...
def semantic_chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """
    Семантически разбивает текст на чанки с помощью SemanticChunker
    """
    # Создаем экземпляр SemanticChunker с заданными эмбеддингами
    chunker = SemanticChunker(chunker_embeddings, breakpoint_threshold_type="interquartile")
    
    # Разбиваем текст на чанки
    chunks = chunker.split_text(text)
    
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
        
        # Очистка текста перед чанкингом (как в пользовательском коде)
        clean_text = re.sub(r'## Страница \d+', '', text_content)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', clean_text)
        clean_text = re.sub(r'\[[^\]]*\]\(.*?\)', '', clean_text)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text.strip())
        
        # Разбиваем очищенный текст на семантические чанки
        chunks = semantic_chunk_text(clean_text, chunk_size)
        
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
    input_dir = Path("/app/data/output")     # было /app/output
    output_dir = Path("/app/data/semantic_chunks")  # было /app/semantic_chunks
    chunk_size = 512  # Приблизительный размер чанка в токенах
    
    print("🚀 Запуск семантического чанкинга")
    print(f"  Входная директория: {input_dir}")
    print(f"  Выходная директория: {output_dir}")
    print(f"  Размер чанка: {chunk_size}")
    
    process_markdown_files(input_dir, output_dir, chunk_size)
    
    print("\n✅ Семантический чанкинг завершен")

if __name__ == "__main__":
    main()