# Advanced RAG Evaluation Report

## Summary

| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|-------------|
| Precision@5 | 0.111 | 0.222 | +100.0% |
| NDCG@10 | 0.986 | 0.913 | -7.4% |
| Context Relevance | 1.333 | 1.667 | +25.0% |

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
   points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressiv...

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines
the g...

