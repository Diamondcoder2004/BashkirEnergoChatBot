# scripts/rag_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import uvicorn
import logging
from typing import List, Optional
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI приложение
app = FastAPI(
    title="RAG API Башкирэнерго",
    description="API для вопросно-ответной системы по документам Башкирэнерго",
    version="1.0.0"
)

# Модели запросов/ответов
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class DocumentResponse(BaseModel):
    content: str
    source: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[DocumentResponse]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    qdrant_available: bool
    models_loaded: bool

# Инициализация моделей
try:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    
    llm = OllamaLLM(
        model="deepseek-r1:8b",
        base_url=OLLAMA_HOST,
        temperature=0.1,
        num_ctx=8192
    )
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url=OLLAMA_HOST
    )
    
    qdrant_client = QdrantClient(host="localhost", port=6333)
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name="bashkir_energo_docs_nomic",
        embeddings=embeddings
    )
    
    logger.info("✅ Модели и векторное хранилище инициализированы")
    MODELS_LOADED = True
except Exception as e:
    logger.error(f"❌ Ошибка инициализации: {e}")
    MODELS_LOADED = False

@app.get("/", summary="Информация о API")
async def root():
    return {
        "message": "RAG API для документов Башкирэнерго",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health - статус сервиса",
            "ask": "/ask - задать вопрос",
            "search": "/search - поиск документов"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Проверка здоровья сервиса")
async def health_check():
    """Проверка доступности всех компонентов системы"""
    try:
        # Проверяем Ollama
        ollama_available = False
        try:
            # Простой запрос к Ollama
            test_response = llm.invoke("test")
            ollama_available = bool(test_response)
        except:
            ollama_available = False
        
        # Проверяем Qdrant
        qdrant_available = False
        try:
            collections = qdrant_client.get_collections()
            qdrant_available = True
        except:
            qdrant_available = False
        
        return HealthResponse(
            status="healthy" if (MODELS_LOADED and ollama_available and qdrant_available) else "degraded",
            ollama_available=ollama_available,
            qdrant_available=qdrant_available,
            models_loaded=MODELS_LOADED
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            ollama_available=False,
            qdrant_available=False,
            models_loaded=False
        )

@app.post("/ask", response_model=AnswerResponse, summary="Задать вопрос системе")
async def ask_question(request: QuestionRequest):
    """Основной endpoint для вопросов к документам Башкирэнерго"""
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"📥 Получен вопрос: {request.question}")
        
        # Поиск релевантных документов
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.top_k}
        )
        
        docs = retriever.invoke(request.question)
        
        if not docs:
            return AnswerResponse(
                question=request.question,
                answer="В документах Башкирэнерго нет информации по этому вопросу",
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Формируем контекст
        context = "\n\n".join([
            f"[Документ: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Промпт для DeepSeek
        prompt = f"""Ты - эксперт по документам Башкирэнерго. Ответь точно на вопрос, используя только предоставленный контекст.

КОНТЕКСТ (документы Башкирэнерго):
{context}

ВОПРОС:
{request.question}

ИНСТРУКЦИИ:
1. Ответь ТОЛЬКО на основе контекста из документов Башкирэнерго
2. Если информации нет в контексте, скажи "В документах Башкирэнерго нет информации по этому вопросу"
3. Будь максимально точным в цифрах, датах и юридических формулировках
4. Для тарифов и нормативов указывай конкретные значения
5. Структурируй ответ, если информация объемная

ОТВЕТ:"""
        
        # Генерация ответа
        answer = llm.invoke(prompt)
        
        # Формируем ответ с источниками
        sources = []
        for doc in docs:
            sources.append(DocumentResponse(
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                source=doc.metadata.get('source', 'Unknown'),
                score=doc.metadata.get('score', 0.0)
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Вопрос обработан за {processing_time:.2f} сек")
        
        return AnswerResponse(
            question=request.question,
            answer=answer.strip(),
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки вопроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки вопроса: {str(e)}")

@app.post("/search", summary="Поиск документов без генерации ответа")
async def search_documents(request: QuestionRequest):
    """Поиск релевантных документов без генерации ответа"""
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.top_k}
        )
        
        docs = retriever.invoke(request.question)
        
        results = []
        for doc in docs:
            results.append(DocumentResponse(
                content=doc.page_content,
                source=doc.metadata.get('source', 'Unknown'),
                score=doc.metadata.get('score', 0.0)
            ))
        
        return {
            "question": request.question,
            "documents_found": len(results),
            "documents": results
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска документов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")

@app.get("/collections", summary="Получить список коллекций")
async def get_collections():
    """Получить информацию о доступных коллекциях в Qdrant"""
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": collections.collections,
            "total": len(collections.collections)
        }
    except Exception as e:
        logger.error(f"❌ Ошибка получения коллекций: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения коллекций: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )