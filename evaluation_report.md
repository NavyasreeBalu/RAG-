# RAG System Evaluation Report

## Performance Summary

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Precision@5 | 0.022 | 0.000 | -100.0% |
| Context Relevance | 1.222 | 1.067 | -12.7% |

## Query Performance Visualization

```
[1] What is the difference between RAG-Sequence and RA...
    Baseline: P@5=0.200 ████░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ ↘↘ -100.0%

[2] How does FlashAttention use tiling to reduce memor...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[3] What is the impact of the rank r hyperparameter in...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[4] Describe the "U-shaped" performance curve observed...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[5] What are the three stages of RLHF used in Instruct...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[6] Which paper first demonstrated emergent abilities ...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[7] How did the Transformer architecture eliminate rec...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[8] Why is LLaMA particularly important for enterprise...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

[9] Compare the bidirectional attention mechanism in B...
    Baseline: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░
    Improved: P@5=0.000 ░░░░░░░░░░░░░░░░░░░░ →  +0.0%

```

## Sample Document Comparison

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

