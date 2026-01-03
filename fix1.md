# Fix 1: Sparse Retrieval Document Format Bug

## Problem Identified
The `sparse_retrieval` function in `improved_approach/rag_pipeline.py` was returning Document objects with inconsistent metadata, causing the hybrid retrieval system to perform worse than baseline.

## Root Cause
```python
# BROKEN CODE (before fix)
def sparse_retrieval(vectorstore, query, k=3):
    all_docs = vectorstore.get()           # Returns raw dict: {'documents': [...], 'metadatas': [...]}
    texts = all_docs['documents']          # Gets raw strings without metadata
    bm25_retriever = BM25Retriever.from_texts(texts)
    return bm25_retriever.invoke(query)    # Returns Documents with EMPTY metadata
```

**Issues:**
1. `vectorstore.get()` returns raw dictionary, not Document objects
2. `BM25Retriever.invoke()` creates Documents with empty metadata (`{}`)
3. Mixed with `dense_retriever()` which returns Documents with full metadata
4. Broke reranker and deduplication logic

## Solution Applied
```python
# FIXED CODE (after fix)
def sparse_retrieval(vectorstore, query, k=3):
    # Get proper Document objects with metadata
    all_docs = vectorstore.similarity_search("", k=1000)
    texts = [doc.page_content for doc in all_docs]
    
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = k
    bm25_results = bm25_retriever.invoke(query)
    
    # Map BM25 results back to original Documents with metadata
    result_docs = []
    for bm25_doc in bm25_results:
        for orig_doc in all_docs:
            if orig_doc.page_content == bm25_doc.page_content:
                result_docs.append(orig_doc)
                break
    
    return result_docs[:k]
```

## Key Changes
1. **Use `similarity_search("", k=1000)`** instead of `get()` to retrieve proper Document objects
2. **Preserve metadata** by mapping BM25 results back to original Documents
3. **Ensure consistency** between dense and sparse retrieval return formats

## Expected Impact
- Hybrid retrieval should now properly combine dense + sparse results
- Reranker will receive consistent Document objects
- Deduplication will work correctly using metadata
- Overall retrieval performance should improve over baseline

## Next Steps
- Re-run evaluation to verify improvement
- Test with multiple queries to confirm fix effectiveness
