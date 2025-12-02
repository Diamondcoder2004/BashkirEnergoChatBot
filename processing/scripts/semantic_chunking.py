"""
Семантический чанкинг документов с улучшенной фильтрацией
"""
import os
import re
import yaml
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from dotenv import load_dotenv

# Настройка кэша моделей
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HOME'] = '/root/.cache/huggingface'

load_dotenv('/app/.env')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

print(f"🚀 Инициализация семантического чанкинга с моделью {EMBEDDING_MODEL}...")

# HuggingFace эмбеддинги с кэшем
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    cache_folder="/root/.cache/huggingface",
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "cuda"}
)

# ... остальной код без изменений ...

def clean_text_content(text: str) -> str:
    """Тщательная очистка текста от мусора"""
    # Удаляем маркеры страниц
    text = re.sub(r'## Страница \d+', '', text)
    # Удаляем изображения
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Удаляем ссылки (оставляем только текст)
    text = re.sub(r'\[([^\]]*)\]\(.*?\)', r'\1', text)
    # Удаляем одиночные цифры и короткие строки
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Удаляем строки короче 10 символов
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Сохраняем только содержательные строки
        if len(stripped) >= 15 and not re.match(r'^[\.\d\s\-–—]*$', stripped):
            clean_lines.append(stripped)
    text = '\n'.join(clean_lines)
    # Убираем множественные переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_meaningful_chunk(text: str, min_length: int = 400) -> bool:
    """Проверяет, является ли чанк содержательным"""
    if len(text) < min_length:
        return False
    
    # Проверяем соотношение текста к общему количеству символов
    text_ratio = len(re.sub(r'\s', '', text)) / len(text)
    if text_ratio < 0.6:  # Слишком много пробелов/переносов
        return False
    
    # Проверяем, что есть достаточно слов
    words = text.split()
    if len(words) < 50:  # Минимум 50 слов
        return False
    
    return True

def semantic_chunk_text(text: str) -> List[str]:
    """
    Семантически разбивает текст на чанки с жесткой фильтрацией
    """
    try:
        chunker = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="interquartile"
        )
        raw_chunks = chunker.split_text(text)
    except Exception as e:
        print(f"⚠️ Ошибка семантического чанкинга: {e}")
        return []
    
    meaningful_chunks = []
    
    for chunk in raw_chunks:
        clean_chunk = clean_text_content(chunk)
        
        # Жесткая проверка на содержательность
        if is_meaningful_chunk(clean_chunk):
            meaningful_chunks.append(clean_chunk)
    
    # Объединяем слишком маленькие соседние чанки
    merged_chunks = []
    current_chunk = ""
    
    for chunk in meaningful_chunks:
        if not current_chunk:
            current_chunk = chunk
        elif len(current_chunk) + len(chunk) < 1500:  # Максимальный размер
            current_chunk += "\n\n" + chunk
        else:
            if is_meaningful_chunk(current_chunk):
                merged_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk and is_meaningful_chunk(current_chunk):
        merged_chunks.append(current_chunk)
    
    return merged_chunks

def process_markdown_files(input_dir: Path, output_dir: Path):
    """Обрабатывает markdown файлы и создает качественные чанки"""
    output_dir.mkdir(exist_ok=True)
    
    # Очищаем предыдущие чанки
    for old_file in output_dir.glob("*.md"):
        old_file.unlink()
    
    md_files = list(input_dir.glob("*.md"))
    print(f"📁 Найдено {len(md_files)} markdown файлов")
    
    total_created = 0
    total_skipped = 0
    
    for md_file in md_files:
        print(f"\n📄 Обработка: {md_file.name}")
        
        try:
            content = md_file.read_text(encoding="utf-8")
            
            # Извлекаем метаданные
            yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
            if yaml_match:
                yaml_header = yaml_match.group(1)
                text_content = yaml_match.group(2)
                try:
                    metadata = yaml.safe_load(yaml_header)
                except yaml.YAMLError:
                    metadata = {"source": md_file.name}
            else:
                text_content = content
                metadata = {"source": md_file.name}
            
            # Очищаем текст
            clean_text = clean_text_content(text_content)
            
            if len(clean_text) < 500:
                print(f"  ⚠️ Слишком короткий текст ({len(clean_text)} символов), пропускаем")
                total_skipped += 1
                continue
            
            # Семантическое разбиение
            chunks = semantic_chunk_text(clean_text)
            
            if not chunks:
                print(f"  ⚠️ Не удалось создать качественные чанки")
                total_skipped += 1
                continue
            
            # Сохраняем чанки
            saved_count = 0
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{md_file.stem}_chunk_{i:03d}.md"
                chunk_path = output_dir / chunk_filename
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "original_file": md_file.name,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk),
                    "word_count": len(chunk.split())
                })
                
                yaml_header = yaml.dump(chunk_metadata, default_flow_style=False, allow_unicode=True)
                chunk_content = f"---\n{yaml_header}---\n\n{chunk}"
                
                chunk_path.write_text(chunk_content, encoding="utf-8")
                saved_count += 1
            
            print(f"  ✅ Сохранено {saved_count} качественных чанков")
            total_created += saved_count
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки файла: {e}")
            total_skipped += 1
    
    print(f"\n📊 ИТОГИ:")
    print(f"   ✅ Создано чанков: {total_created}")
    print(f"   ⚠️ Пропущено файлов: {total_skipped}")

def main():
    input_dir = Path("/app/data/output")
    output_dir = Path("/app/data/semantic_chunks")
    
    print("🚀 Запуск улучшенного семантического чанкинга")
    print(f"  📁 Вход: {input_dir}")
    print(f"  📁 Выход: {output_dir}")
    
    if not input_dir.exists():
        print(f"❌ Входная директория не существует: {input_dir}")
        return
    
    process_markdown_files(input_dir, output_dir)
    print("\n✅ Семантический чанкинг завершен")

if __name__ == "__main__":
    main()