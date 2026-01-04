# RAG System Comparison Report

## Summary

- **Total queries evaluated**: 9
- **Average baseline time**: 0.04s
- **Average improved time**: 2.22s
- **Average speed improvement**: -6575.3%

## Detailed Comparison

| Query | Baseline Time | Improved Time | Speed Improvement | Baseline Sources | Improved Sources |
|-------|---------------|---------------|-------------------|------------------|------------------|
| What is the difference between RAG-Seque... | 0.03s | 4.64s | +-16848.2% | 3 | 3 |
| How does FlashAttention use tiling to re... | 0.05s | 1.71s | +-3073.1% | 3 | 3 |
| What is the impact of the rank r hyperpa... | 0.05s | 2.30s | +-4897.9% | 3 | 3 |
| Describe the "U-shaped" performance curv... | 0.03s | 1.71s | +-5411.9% | 3 | 3 |
| What are the three stages of RLHF used i... | 0.04s | 2.07s | +-5350.7% | 3 | 3 |
| Which paper first demonstrated emergent ... | 0.03s | 1.64s | +-6057.9% | 3 | 3 |
| How did the Transformer architecture eli... | 0.05s | 1.64s | +-3475.4% | 3 | 3 |
| Why is LLaMA particularly important for ... | 0.03s | 1.43s | +-4951.0% | 3 | 3 |
| Compare the bidirectional attention mech... | 0.03s | 2.81s | +-9111.4% | 3 | 3 |

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

