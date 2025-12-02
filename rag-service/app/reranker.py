# reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import logging
import os
import torch
import numpy as np
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L6-v2')
RERANK_THRESHOLD = float(os.getenv('RERANK_THRESHOLD', '0.1'))

try:
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=512
    )
    logger.info(f"✅ Reranker model '{RERANKER_MODEL}' loaded successfully on {reranker.model.device}")
except Exception as e:
    logger.error(f"❌ Failed to load reranker '{RERANKER_MODEL}': {e}")
    reranker = None

def rerank_docs(query: str, documents: List[str]) -> Dict:
    if not documents:
        return {'results': []}
    
    if reranker is None:
        return _fallback_rerank(query, documents)
    
    try:
        # Обрабатываем батчами для экономии памяти
        batch_size = 8
        all_scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            pairs = [(query, doc) for doc in batch_docs]
            
            with torch.no_grad():
                batch_scores = reranker.predict(
                    pairs, 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            
            if torch.is_tensor(batch_scores):
                batch_scores = batch_scores.cpu().numpy()
            
            all_scores.extend(batch_scores)
        
        scores = np.array(all_scores)
        
        logger.info(f"📊 Raw scores range: [{scores.min():.3f}, {scores.max():.3f}] mean: {scores.mean():.3f}")
        
        # НОРМАЛИЗАЦИЯ scores в диапазон [0, 1]
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = np.ones_like(scores)  # все одинаковые
        
        logger.info(f"📊 Normalized scores range: [{normalized_scores.min():.3f}, {normalized_scores.max():.3f}]")
        
        reranked_results = []
        for idx, score in enumerate(normalized_scores):
            reranked_results.append({
                'index': idx,
                'relevance_score': float(score)
            })
        
        return {'results': reranked_results}
    
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        return _fallback_rerank(query, documents)

def _fallback_rerank(query: str, documents: List[str]) -> Dict:
    import re
    
    query_terms = set(re.findall(r'\w+', query.lower()))
    
    reranked_results = []
    for idx, doc in enumerate(documents):
        doc_terms = set(re.findall(r'\w+', doc.lower()))
        common_terms = query_terms.intersection(doc_terms)
        
        score = len(common_terms) / len(query_terms) if query_terms else 0.0
        
        reranked_results.append({
            'index': idx,
            'relevance_score': min(score, 1.0)
        })
    
    return {'results': reranked_results}

def get_relevant_docs(query: str, documents: List[str], threshold: float = None) -> List[Dict]:
    """Gets relevant documents by re-ranking and filtering based on threshold."""
    if not documents:
        return []
    
    if threshold is None:
        threshold = RERANK_THRESHOLD
    
    answer = rerank_docs(query=query, documents=documents)
    sorted_results = sorted(answer['results'], key=lambda x: x['relevance_score'], reverse=True)
    
    logger.info(f"🎯 Applying threshold: {threshold}")
    relevant_results = [r for r in sorted_results if r['relevance_score'] > threshold]

    # Логируем топ-5 scores для отладки
    top_scores = [f"{r['relevance_score']:.3f}" for r in sorted_results[:5]]
    logger.info(f"🏆 Top-5 scores: {', '.join(top_scores)}")

    if len(relevant_results) < 3:
        remaining = [r for r in sorted_results if r not in relevant_results]
        needed = 3 - len(relevant_results)
        relevant_results.extend(remaining[:needed])
        logger.info(f"📈 Extended to {len(relevant_results)} documents")

    relevant_docs_with_scores = []
    for r in relevant_results:
        relevant_docs_with_scores.append({
            'content': documents[r['index']],
            'relevance_score': r['relevance_score'],
            'index': r['index']
        })
    
    return relevant_docs_with_scores

def get_relevant_chunks(query: str, vector_store, k: int = 10) -> List[str]:
    """Получает релевантные чанки из векторного хранилища."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})  # <-- используем переданный параметр k
    docs = retriever.invoke(query)
    texts = [doc.page_content for doc in docs]
    logger.info(f"🔍 Извлечено {len(texts)} документов с k={k}")
    return texts