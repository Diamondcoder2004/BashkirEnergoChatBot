from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import uvicorn
import logging
from typing import List, Optional
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные константы
COLLECTION_NAME = "bashkir_energo_rubert"  # Название коллекции

# Создаем приложение
app = FastAPI(
    title="RAG API Башкирэнерго",
    description="API для вопросно-ответной системы по документам Башкирэнерго",
    version="1.0.0"
)

# Модели данных
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class DocumentResponse(BaseModel):
    content: str
    source: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[DocumentResponse]

class HealthResponse(BaseModel):
    status: str
    service: str
    collection: str

# Глобальные переменные для моделей
llm = None
vector_store = None
SERVICE_READY = False


def compress(text: str) -> str:
    """
    Compresses text to 20% of its original volume while preserving key facts.
    
    Args:
        text: Input text to compress
        
    Returns:
        Compressed text
    """
    return llm.invoke(f"Сожми текст до 20% объёма, сохрани ключевые факты:\n\n{text}")

@app.on_event("startup")
async def startup_event():
    global llm, vector_store, SERVICE_READY
    try:
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
        
        logger.info(f"🔄 Инициализация Ollama: {OLLAMA_HOST}")
        logger.info(f"🔄 Инициализация Qdrant: {QDRANT_HOST}")
        logger.info(f"📁 Используемая коллекция: {COLLECTION_NAME}")
        
        # Инициализация LLM
        llm = OllamaLLM(
            model="bambucha/saiga-llama3",
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_ctx=8192,
            timeout=120
        )
        
        # Инициализация русскоязычных эмбеддингов с использованием HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="MiniLM-L12-v2",  
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Инициализация Qdrant
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
        
        # Проверяем доступность коллекции
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        logger.info(f"📚 Доступные коллекции: {collection_names}")
        
        if COLLECTION_NAME not in collection_names:
            logger.warning(f"❌ Коллекция '{COLLECTION_NAME}' не найдена")
            SERVICE_READY = False
            return
        
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        
        # Тестовый запрос к Qdrant
        test_results = vector_store.similarity_search("подключение электричество", k=1)
        logger.info(f"✅ Тестовый поиск: найдено {len(test_results)} документов")
        
        # Тестовый запрос к Ollama
        test_response = llm.invoke("Ответь 'OK'")
        logger.info(f"✅ Тест Ollama: {test_response.strip()}")
        
        logger.info("✅ RAG сервис полностью инициализирован")
        SERVICE_READY = True
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG: {e}")
        SERVICE_READY = False

@app.get("/")
async def root():
    return {
        "message": "RAG API Башкирэнерго", 
        "status": "running",
        "collection": COLLECTION_NAME
    }

@app.get("/health")
async def health():
    return HealthResponse(
        status="ready" if SERVICE_READY else "degraded", 
        service="rag-api",
        collection=COLLECTION_NAME
    )

@app.get("/test")
async def test():
    return {
        "test": "ok", 
        "ready": SERVICE_READY,
        "collection": COLLECTION_NAME
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not SERVICE_READY:
        return AnswerResponse(
            question=request.question,
            answer=f"Сервис не готов. Коллекция '{COLLECTION_NAME}' не найдена.",
            sources=[]
        )
    
    try:
        logger.info(f"📥 Вопрос: {request.question}")
        logger.info(f"📁 Поиск в коллекции: {COLLECTION_NAME}")
        
        # Поиск релевантных документов
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.top_k}
        )
        
        docs = retriever.invoke(request.question)
        logger.info(f"🔍 Найдено документов: {len(docs)}")
        
        if not docs:
            return AnswerResponse(
                question=request.question,
                answer=f"В коллекции '{COLLECTION_NAME}' нет информации по этому вопросу",
                sources=[]
            )
        
        # Формируем контекст
        context = "\n\n".join([
            f"[Документ: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        logger.info(f"📄 Контекст подготовлен ({len(context)} символов)")
        
        # Сжимаем контекст перед подачей в LLM
        compressed_context = compress(context)
        logger.info(f"📦 Контекст сжат до ({len(compressed_context)} символов)")
        
        # Промпт для Saiga-llama3
        prompt = f"""Ты - AI-ассистент по документам Башкирэнерго. Проанализируй контекст и ответь на вопрос.

КОНТЕКСТ (документы Башкирэнерго):
{compressed_context}

ВОПРОС:
{request.question}

ИНСТРУКЦИИ:
1. ВНИМАТЕЛЬНО проанализируй контекст на наличие информации по вопросу
2. Если информация ЕСТЬ в контексте - дай развернутый ответ с конкретными деталями
3. Если информации НЕТ - честно скажи "В предоставленных документах нет информации по этому вопросу"
4. Будь максимально точным: указывай конкретные цифры, сроки, требования, тарифы
5. Используй только информацию из контекста, не придумывай
6. Если находишь несколько вариантов - перечисли их все

ВАЖНО: Не говори что информации нет, если она есть в контексте!

ОТВЕТ:"""
        
        # Генерация ответа
        logger.info("🤖 Генерация ответа с Saiga-llama3...")
        answer = llm.invoke(prompt)
        logger.info(f"✅ Ответ сгенерирован")
        
        # Формируем источники
        sources = []
        for doc in docs:
            sources.append(DocumentResponse(
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                source=doc.metadata.get('source', 'Unknown')
            ))
        
        return AnswerResponse(
            question=request.question,
            answer=answer.strip(),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки вопроса: {e}")
        return AnswerResponse(
            question=request.question,
            answer=f"Ошибка обработки: {str(e)}",
            sources=[]
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )