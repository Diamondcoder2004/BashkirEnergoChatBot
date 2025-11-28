from langchain_community.llms import Ollama
from app.config import OLLAMA_HOST
from typing import List, Dict, Optional

def rerank_docs(query: str, documents: List[str]) -> Dict:
    """
    Re-ranks documents based on relevance to the query using a local reranker model.
    
    Args:
        query: The search query
        documents: List of document texts to re-rank
        
    Returns:
        Dictionary with results containing relevance scores
    """
    # Initialize Ollama LLM with the Qwen3-Reranker-0.6B model
    llm = Ollama(
        model="dengcao/Qwen3-Reranker-0.6B:latest",
        base_url=OLLAMA_HOST,
        temperature=0  # For consistent reranking
    )
    
    # Create reranking results
    reranked_results = []
    for idx, doc in enumerate(documents):
        # Create a prompt that asks the model to score the relevance
        prompt = f"""Оцените релевантность документа к запросу. 
        Присвойте оценку от 0 до 1, где 1 - максимально релевантно.
        
        Запрос: {query}
        
        Документ: {doc}
        
        Оценка релевантности (число от 0 до 1):"""
        
        try:
            response = llm.invoke(prompt)
            # Extract score from response (assuming it returns a number)
            response_text = response.strip()
            # Try to extract a float number from the response
            score = 0.0
            for word in response_text.split():
                word_clean = word.replace(',', '.').strip('.,;:')
                if word_clean.replace('.', '').isdigit():
                    score = float(word_clean)
                    break
        except:
            score = 0.0  # Default score if parsing fails
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        reranked_results.append({
            'index': idx,
            'relevance_score': score
        })
    
    return {'results': reranked_results}


def get_relevant_docs(query: str, documents: List[str], threshold: float) -> List[str]:
    """
    Gets relevant documents by re-ranking and filtering based on threshold.
    
    Args:
        query: The search query
        documents: List of document texts
        threshold: Minimum relevance score threshold
        
    Returns:
        List of relevant document texts
    """
    answer = rerank_docs(query=query, documents=documents)

    # Sort results by relevance score in descending order
    sorted_results = sorted(answer['results'], key=lambda x: x['relevance_score'], reverse=True)

    # Filter by threshold
    relevant_results = [r for r in sorted_results if r['relevance_score'] > threshold]

    # If less than 3 chunks pass the threshold, add best of the remaining
    if len(relevant_results) < 3:
        remaining = [r for r in sorted_results if r not in relevant_results]
        # Take as many as needed to reach 3
        needed = 3 - len(relevant_results)
        relevant_results.extend(remaining[:needed])

    # Extract texts
    relevant_texts = [documents[r['index']] for r in relevant_results]

    if relevant_texts:
        return relevant_texts
    else:
        print(f"No documents (even >= 3) for query: {query}")
        return []


def get_relevant_chunks(query: str, vector_store) -> List[str]:
    """
    Retrieves relevant chunks from vector store.
    
    Args:
        query: The search query
        vector_store: Vector store instance
        
    Returns:
        List of relevant document texts
    """
    # Get only texts (not Document objects)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Get more to re-rank
    docs = retriever.invoke(query)
    texts = [doc.page_content for doc in docs]  # Extract text
    return texts