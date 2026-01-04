# Advanced RAG Evaluation Report

## Summary

| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Precision@5 | 0.178 | 0.244 | +37.5% |
| NDCG@10 | 0.958 | 0.900 | -6.1% |
| Context Relevance | 1.533 | 1.733 | +13.0% |

## Sample Document Comparison (Before vs After)

**Query**: What is the difference between RAG-Sequence and RAG-Token?

### Baseline Retrieved Documents:
1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...

2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   minimize the negative marginal log-likelihood of each target, ∑
j−log p(yj|xj) using stochastic
grad...

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressiv...

### Improved Retrieved Documents:
1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...

2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   in 71% of cases, and a gold article is present in the top 10 retrieved articles in 90% of cases.
4.5...

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressiv...

