# main.py
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
import sys
import time
import gc
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from reranker import get_relevant_docs, get_relevant_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bashkir_energo_minilm_v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "bambucha/saiga-llama3")

app = FastAPI(
    title="RAG API Башкирэнерго",
    description="API для вопросно-ответной системы по документам Башкирэнерго",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    temperature: Optional[float] = 0.1
    rerank_threshold: Optional[float] = 0.1

class DocumentResponse(BaseModel):
    content: str
    source: str
    relevance_score: Optional[float] = None

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[DocumentResponse]
    parameters: dict

class HealthResponse(BaseModel):
    status: str
    service: str
    collection: str

class ClearCacheResponse(BaseModel):
    status: str
    message: str
    gpu_cleared: bool

llm = None
vector_store = None
SERVICE_READY = False

def clear_gpu_memory():
    """Очистка памяти GPU если доступно"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 GPU memory cleared")
            return True
    except ImportError:
        logger.warning("⚠️ torch not available for GPU memory clearing")
    except Exception as e:
        logger.warning(f"⚠️ GPU memory clearing failed: {e}")
    
    # Всегда делаем garbage collection
    gc.collect()
    return False

@app.post("/clear-cache")
async def clear_cache():
    """
    Очистка кэша GPU и памяти
    """
    try:
        gpu_cleared = clear_gpu_memory()
        
        # Дополнительная очистка для LangChain если нужно
        try:
            from langchain.globals import set_llm_cache
            set_llm_cache(None)
            logger.info("🧹 LangChain cache cleared")
        except:
            pass
            
        return ClearCacheResponse(
            status="success",
            message="Cache cleared successfully",
            gpu_cleared=gpu_cleared
        )
    except Exception as e:
        logger.error(f"❌ Cache clearing error: {e}")
        return ClearCacheResponse(
            status="error",
            message=f"Cache clearing failed: {str(e)}",
            gpu_cleared=False
        )

def compress(text: str) -> str:
    try:
        return llm.invoke(f"Сожми текст до 20% объёма, сохрани ключевые факты:\n\n{text}")
    except Exception as e:
        logger.error(f"Ошибка сжатия текста: {e}")
        return text

@app.on_event("startup")
async def startup_event():
    global llm, vector_store, SERVICE_READY
    try:
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        QDRANT_HOST = os.getenv("QDRANT_HOST", "host.docker.internal")
        
        logger.info(f"🔄 Инициализация Ollama: {OLLAMA_HOST}")
        logger.info(f"🔄 Инициализация Qdrant: {QDRANT_HOST}")
        logger.info(f"📁 Используемая коллекция: {COLLECTION_NAME}")
        logger.info(f"🤖 Модель LLM: {OLLAMA_MODEL}")
        
        time.sleep(5)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                llm = OllamaLLM(
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=0.1,
                    num_ctx=8192,
                    timeout=120
                )
                test_response = llm.invoke("Ответь 'OK'")
                logger.info(f"✅ Тест Ollama: {test_response.strip()}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Попытка {attempt + 1}/{max_retries} не удалась: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="/root/.cache/huggingface",
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": "cuda"}
            )
            logger.info("✅ Эмбеддинги инициализированы")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации эмбеддингов: {e}")
            SERVICE_READY = False
            return
        
        try:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333, timeout=60)
            
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
            
            test_results = vector_store.similarity_search("подключение электричество", k=1)
            logger.info(f"✅ Тестовый поиск: найдено {len(test_results)} документов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Qdrant: {e}")
            SERVICE_READY = False
            return
        
        logger.info("✅ RAG сервис полностью инициализирован")
        SERVICE_READY = True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка инициализации RAG: {e}")
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

@app.post("/reset")
async def reset_llm():
    global llm
    try:
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_ctx=8192,
            timeout=120
        )
        return {"status": "LLM reset successful"}
    except Exception as e:
        return {"status": f"Error: {str(e)}"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not SERVICE_READY:
        return AnswerResponse(
            question=request.question,
            answer=f"Сервис не готов. Коллекция '{COLLECTION_NAME}' не найдена или сервис не инициализирован.",
            sources=[],
            parameters={}
        )
    
    try:
        logger.info(f"📥 Вопрос: {request.question}")
        logger.info(f"🎛️ Параметры: top_k={request.top_k}, temperature={request.temperature}, rerank_threshold={request.rerank_threshold}")
        
        global llm
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=request.temperature,
            num_ctx=8192,
            timeout=120
        )
        
        # Используем top_k для начального поиска, но умножаем на 3 для реранкинга
        initial_search_k = max(request.top_k * 3, 10)  # ищем больше документов для качественного реранкинга
        initial_docs = get_relevant_chunks(request.question, vector_store, k=initial_search_k)
        logger.info(f"🔍 Найдено начальных документов: {len(initial_docs)} (k={initial_search_k})")
        
        if not initial_docs:
            return AnswerResponse(
                question=request.question,
                answer="В базе документов нет информации по этому вопросу",
                sources=[],
                parameters={
                    "top_k": request.top_k,
                    "temperature": request.temperature,
                    "rerank_threshold": request.rerank_threshold,
                    "initial_search_k": initial_search_k
                }
            )
        
        relevant_docs_with_scores = get_relevant_docs(request.question, initial_docs, request.rerank_threshold)
        logger.info(f"✅ Отранжировано релевантных документов: {len(relevant_docs_with_scores)}")
        
        for i, doc in enumerate(relevant_docs_with_scores[:5]):  # показываем только топ-5
            logger.info(f"📊 Документ {i+1}: score={doc['relevance_score']:.3f}")
        
        relevant_docs_with_scores = relevant_docs_with_scores[:request.top_k]
        
        if not relevant_docs_with_scores:
            return AnswerResponse(
                question=request.question,
                answer="Не найдено достаточно релевантной информации по данному вопросу",
                sources=[],
                parameters={
                    "top_k": request.top_k,
                    "temperature": request.temperature,
                    "rerank_threshold": request.rerank_threshold,
                    "initial_search_k": initial_search_k
                }
            )
        
        context = "\n\n".join([doc['content'] for doc in relevant_docs_with_scores])
        logger.info(f"📄 Контекст подготовлен ({len(context)} символов)")
        
        compressed_context = compress(context)
        logger.info(f"📦 Контекст сжат до ({len(compressed_context)} символов)")
        
        prompt = f"""Ты - AI-ассистент по документам Башкирэнерго. Ответь на вопрос на основе предоставленного контекста.

КОНТЕКСТ:
{compressed_context}

ВОПРОС:
{request.question}

ИНСТРУКЦИИ:
1. Используй только информацию из контекста
2. Если информации нет - скажи "В предоставленных документах нет информации по этому вопросу"
3. Будь точным и конкретным
4. Не придумывай информацию

ОТВЕТ:"""
        
        logger.info("🤖 Генерация ответа...")
        answer = llm.invoke(prompt)
        logger.info(f"✅ Ответ сгенерирован")
        
        sources = []
        for i, doc in enumerate(relevant_docs_with_scores):
            sources.append(DocumentResponse(
                content=doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                source=f"Документ {i+1}",
                relevance_score=round(doc['relevance_score'], 3)
            ))
        
        return AnswerResponse(
            question=request.question,
            answer=answer.strip(),
            sources=sources,
            parameters={
                "top_k": request.top_k,
                "temperature": request.temperature,
                "rerank_threshold": request.rerank_threshold,
                "initial_search_k": initial_search_k
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки вопроса: {e}")
        return AnswerResponse(
            question=request.question,
            answer=f"Произошла ошибка при обработке запроса: {str(e)}",
            sources=[],
            parameters={
                "top_k": request.top_k,
                "temperature": request.temperature,
                "rerank_threshold": request.rerank_threshold,
                "initial_search_k": initial_search_k
            }
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )