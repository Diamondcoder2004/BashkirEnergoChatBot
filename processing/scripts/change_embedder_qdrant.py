# scripts/change_embedder_qdrant.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
import pathlib
import re
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_collection():
    """Создает новую коллекцию с rubert эмбеддером"""
    
    # Новый эмбеддер
    embeddings = OllamaEmbeddings(
        model="ognivo777/rubert-mini-frida:latest",
        base_url="http://host.docker.internal:11434"
    )
    
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    # Загружаем чанки
    path = pathlib.Path("/app/data/semantic_chunks")
    documents = []
    
    for md_file in path.glob("*.md"):
        logger.info(f"📄 Загрузка: {md_file.name}")
        
        try:
            content = md_file.read_text(encoding="utf-8")
            
            # Парсим YAML заголовок
            match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
            if match:
                yaml_header = match.group(1)
                text_content = match.group(2)
                try:
                    metadata = yaml.safe_load(yaml_header)
                except:
                    metadata = {}
            else:
                text_content = content
                metadata = {}
            
            documents.append(Document(
                page_content=text_content,
                metadata={**metadata, "source": md_file.name}
            ))
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {md_file.name} - {e}")
    
    logger.info(f"📚 Загружено {len(documents)} документов")
    
    # Создаем новую коллекцию
    NEW_COLLECTION_NAME = "bashkir_energo_rubert"
    
    Qdrant.from_documents(
        documents,
        embeddings,
        location="http://localhost:6333",
        collection_name=NEW_COLLECTION_NAME,
        force_recreate=True
    )
    
    logger.info(f"✅ Новая коллекция '{NEW_COLLECTION_NAME}' создана с rubert-mini-frida!")

if __name__ == "__main__":
    reindex_collection()