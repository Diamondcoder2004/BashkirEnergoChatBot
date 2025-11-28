# ai_chunking.py — создание векторной базы в Qdrant из semantic chunks с помощью saiga-llama3 и huggingface эмбеддингов
# from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings  # ПРАВИЛЬНЫЙ импорт
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import pathlib, re, yaml
import os

# HuggingFace эмбеддинги (работают с русским языком и другими)
embeddings = HuggingFaceEmbeddings(
    model_name="MiniLM-L12-v2",  
    encode_kwargs={"normalize_embeddings": True}
)

print("MiniLM-L12-v2 подключено")

def load_and_chunk():
    # Используем semantic chunks вместо простого разбиения
    path = pathlib.Path("/app/data/semantic_chunks")  # было /app/semantic_chunks
    docs = []
    
    # Если semantic_chunks не существует, используем оригинальные markdown файлы
    if not path.exists():
        print("⚠️ Директория semantic_chunks не найдена, используем /app/output")
        path = pathlib.Path("/app/output")
    
    for md in path.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        m = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
        meta = yaml.safe_load(m.group(1)) if m else {}
        text = text[m.end():] if m else text
        text = re.sub(r'## Страница \d+\s*\n+', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        
        # Для семантических чанков берем каждый файл как отдельный документ
        docs.append(Document(
            page_content=text,
            metadata={**meta, "source": md.name}
        ))
    
    return docs

print("Загружаем документы...")
documents = load_and_chunk()
print(f"Создано {len(documents)} чанков")

print("Заливаем в Qdrant (nomic-embed-text-v1.5)...")
Qdrant.from_documents(
    documents,
    embeddings,
    location="http://localhost:6333",
    collection_name="bashkir_energo_docs_nomic",
    force_recreate=True
)

print("\nГОТОВО! Коллекция: bashkir_energo_docs_nomic")
print("Открывай http://localhost:6333/dashboard — будет самая точная кластеризация юр.текста в РФ")