from langchain_community.cross_encoders import CrossEncoder
from typing import List, Dict

# Initialize Russian BERT-based reranker
reranker = CrossEncoder("DeepPavlov/rubert-base-cased-ranker")

def rerank_docs(query: str, documents: List[str]) -> Dict:
    """
    Re-ranks documents based on relevance to the query using a Russian BERT-based reranker.
    
    Args:
        query: The search query
        documents: List of document texts to re-rank
        
    Returns:
        Dictionary with results containing relevance scores
    """
    # Use the cross-encoder to score query-document pairs
    scores = reranker.predict([(query, doc) for doc in documents])
    
    # Create reranking results
    reranked_results = []
    for idx, score in enumerate(scores):
        # Convert to float and ensure score is between 0 and 1
        score_float = float(score)
        score_float = max(0.0, min(1.0, score_float))
        
        reranked_results.append({
            'index': idx,
            'relevance_score': score_float
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