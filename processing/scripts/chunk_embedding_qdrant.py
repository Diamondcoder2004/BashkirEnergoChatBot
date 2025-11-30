"""
Создание векторной базы из семантических чанков
"""
import os
import re
import yaml
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from dotenv import load_dotenv

# Настройка кэша моделей
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HOME'] = '/root/.cache/huggingface'

load_dotenv('/app/.env')

# Получаем настройки из .env
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'bashkir_energo_docs')
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

print(f"🚀 Инициализация создания векторной базы с моделью {EMBEDDING_MODEL}...")

# HuggingFace эмбеддинги с кэшем
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    cache_folder="/root/.cache/huggingface",
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"device": "cuda"}
)

# ... остальной код без изменений ...

def filter_quality_chunks(documents: list) -> list:
    """Фильтрует чанки по качеству"""
    quality_docs = []
    skipped_count = 0
    
    for doc in documents:
        text = doc.page_content
        
        # Проверяем качество чанка
        if (len(text) >= 400 and 
            len(text.split()) >= 60 and  # Минимум 60 слов
            not re.search(r'^\s*\d+\s*$', text) and  # Не только цифры
            not re.search(r'^[\.\-\s]*$', text)):  # Не только точки и тире
            
            # Очищаем текст перед сохранением
            clean_text = re.sub(r'\n{3,}', '\n\n', text.strip())
            doc.page_content = clean_text
            quality_docs.append(doc)
        else:
            skipped_count += 1
    
    print(f"📊 Фильтрация: {len(quality_docs)} качественных чанков, {skipped_count} отброшено")
    return quality_docs

def load_semantic_chunks() -> list:
    """Загружает семантические чанки"""
    chunks_dir = Path("/app/data/semantic_chunks")
    
    if not chunks_dir.exists():
        print(f"❌ Директория семантических чанков не найдена: {chunks_dir}")
        return []
    
    md_files = list(chunks_dir.glob("*.md"))
    print(f"📁 Найдено {len(md_files)} семантических чанков")
    
    documents = []
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            
            # Извлекаем метаданные и контент
            yaml_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
            if yaml_match:
                yaml_header = yaml_match.group(1)
                text_content = yaml_match.group(2)
                
                try:
                    metadata = yaml.safe_load(yaml_header)
                except yaml.YAMLError:
                    metadata = {}
                
                # Очищаем текст
                clean_text = re.sub(r'\n{3,}', '\n\n', text_content.strip())
                
                documents.append(Document(
                    page_content=clean_text,
                    metadata={
                        **metadata,
                        "source_file": md_file.name,
                        "chunk_size": len(clean_text)
                    }
                ))
                
        except Exception as e:
            print(f"⚠️ Ошибка загрузки чанка {md_file.name}: {e}")
    
    return documents

def create_vector_store():
    """Создает векторное хранилище в Qdrant"""
    print("📥 Загрузка семантических чанков...")
    documents = load_semantic_chunks()
    
    if not documents:
        print("❌ Нет документов для обработки")
        return
    
    print(f"✅ Загружено {len(documents)} чанков")
    
    # Фильтруем по качеству
    quality_documents = filter_quality_chunks(documents)
    
    if not quality_documents:
        print("❌ Нет качественных чанков после фильтрации")
        return
    
    print(f"🎯 Создание векторной базы из {len(quality_documents)} чанков...")
    
    try:
        # Создаем векторное хранилище
        vector_store = Qdrant.from_documents(
            documents=quality_documents,
            embedding=embeddings,
            url=f"http://{QDRANT_HOST}:6333",
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        
        print(f"✅ Векторная база создана успешно!")
        print(f"   📊 Коллекция: {COLLECTION_NAME}")
        print(f"   📈 Векторов: {len(quality_documents)}")
        print(f"   🔗 Qdrant: http://{QDRANT_HOST}:6333/dashboard")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Ошибка создания векторной базы: {e}")
        return None

def main():
    print(f"🔧 Настройки:")
    print(f"   Модель: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Коллекция: {COLLECTION_NAME}")
    print(f"   Qdrant: {QDRANT_HOST}:6333")
    
    vector_store = create_vector_store()
    
    if vector_store:
        print("\n🎉 Векторная база готова к использованию!")
    else:
        print("\n💥 Не удалось создать векторную базу")

if __name__ == "__main__":
    main()