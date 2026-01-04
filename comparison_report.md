# RAG System Comparison Report

## Summary

- **Total queries evaluated**: 9
- **Average baseline time**: 8.20s
- **Average improved time**: 10.06s
- **Average speed improvement**: -381.5%

## Detailed Comparison

| Query | Baseline Time | Improved Time | Speed Improvement | Baseline Sources | Improved Sources |
|-------|---------------|---------------|-------------------|------------------|------------------|
| What is the difference between RAG-Seque... | 0.62s | 5.01s | +-706.6% | 3 | 3 |
| How does FlashAttention use tiling to re... | 0.75s | 1.65s | +-119.7% | 3 | 3 |
| What is the impact of the rank r hyperpa... | 0.51s | 13.69s | +-2590.1% | 3 | 3 |
| Describe the "U-shaped" performance curv... | 8.81s | 10.44s | +-18.5% | 3 | 3 |
| What are the three stages of RLHF used i... | 15.25s | 12.60s | +17.4% | 3 | 3 |
| Which paper first demonstrated emergent ... | 14.91s | 11.11s | +25.4% | 3 | 3 |
| How did the Transformer architecture eli... | 7.39s | 10.28s | +-39.0% | 3 | 3 |
| Why is LLaMA particularly important for ... | 12.95s | 12.57s | +2.9% | 3 | 3 |
| Compare the bidirectional attention mech... | 12.59s | 13.23s | +-5.0% | 3 | 3 |

## Sample Retrieval Comparison

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
pθ(yi|x,z,y 1:i−1)
RAG-Token Model In the RAG-Token model we ...

2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   minimize the negative marginal log-likelihood of each target, ∑
j−log p(yj|xj) using stochastic
gradient descent with Adam [28]. Updating the document...

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressive given that (i) those models access gold passages...

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
pθ(yi|x,z,y 1:i−1)
RAG-Token Model In the RAG-Token model we ...

2. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressive given that (i) those models access gold passages...

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf**
   minimize the negative marginal log-likelihood of each target, ∑
j−log p(yj|xj) using stochastic
gradient descent with Adam [28]. Updating the document...

