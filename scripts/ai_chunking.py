# ai_chunking.py — окончательная версия с frida-bert через .env
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import pathlib, re, yaml

# DeepSeek-R1-8b (на хосте)
llm = OllamaLLM(model="deepseek-r1:8b",
                base_url="http://host.docker.internal:11434",
                temperature=0.1)

# Твой приватный rubert-mini-frida (токен берётся из .env → HUGGINGFACE_HUB_TOKEN)
embeddings = HuggingFaceEmbeddings(
    model_name="ognivo777/rubert-mini-frida",
    model_kwargs={"device": "cuda"}
)

print("DeepSeek-R1-8b + rubert-mini-frida — всё подключено!")

def load_and_chunk():
    path = pathlib.Path("/app/output")
    docs = []
    for md in path.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        m = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
        meta = yaml.safe_load(m.group(1)) if m else {}
        text = text[m.end():] if m else text
        text = re.sub(r'## Страница \d+\s*\n+', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text.strip())

        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 100]
        for i, c in enumerate(chunks):
            docs.append(Document(
                page_content=c,
                metadata={**meta, "source": md.name, "chunk_id": i}
            ))
    return docs

print("Загружаем документы...")
documents = load_and_chunk()
print(f"Создано {len(documents)} чанков")

print("Заливаем в Qdrant (frida-bert)...")
Qdrant.from_documents(
    documents,
    embeddings,
    location="http://localhost:6333",
    collection_name="bashkir_energo_docs_frida",
    force_recreate=True
)

print("\nГОТОВО! Коллекция: bashkir_energo_docs_frida")
print("Открывай http://localhost:6333/dashboard — будет самая точная кластеризация юр.текста в РФ")