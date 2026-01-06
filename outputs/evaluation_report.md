# RAG Evaluation Report

## Executive Summary
**Baseline Precision@5:** 0.29
**Hybrid Precision@5:** 0.40
**P@5 Improvement:** +40.0%

**Baseline NDCG@5:** 0.94
**Hybrid NDCG@5:** 0.94
**NDCG@5 Improvement:** -0.2%

## Detailed Query Results
### Q: What is the difference between RAG-Sequence and RAG-Token?

**Metrics:**
- Baseline P@5: 0.40, NDCG@5: 0.94
- Hybrid P@5: 0.20, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: minimize the negative marginal log-likelihood of each target, ∑
j−log p(yj|xj) using stochastic
grad...
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressiv...

**Hybrid Retrieved Docs:**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: in 71% of cases, and a gold article is present in the top 10 retrieved articles in 90% of cases.
4.5...
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is
impressiv...

**Baseline Answer:**
The main difference between RAG-Sequence and RAG-Token lies in how they generate sequences and how they approximate the arg max of y given x.

RAG-Token can be viewed as a standard, autoregressive seq...

**Hybrid Answer:**
The main difference between RAG-Sequence and RAG-Token lies in their decoding mechanisms and how they approximate the arg max of the probability distribution p(y|x).

RAG-Token can be seen as a standa...

---

### Q: What is the impact of the rank 'r' hyperparameter in LoRA?

**Metrics:**
- Baseline P@5: 0.80, NDCG@5: 1.00
- Hybrid P@5: 0.80, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: greater efﬁciency (AdapterD). We cite numbers from prior works whenever possible to maximize
the num...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
r(Y...
3. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: Table 5: Validation accuracy on WikiSQL and MultiNLI after applying LoRA to different types of
atten...

**Hybrid Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
r(Y...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: Table 5: Validation accuracy on WikiSQL and MultiNLI after applying LoRA to different types of
atten...
3. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: rank as small as one sufﬁces for adapting bothWq and Wv on these datasets while trainingWq alone
nee...

**Baseline Answer:**
The impact of the rank 'r' hyperparameter in LoRA is that an unexpectedly small rank (as small as one) can suffice for adapting both Wq and Wv on certain datasets, while training Wq alone needs a larg...

**Hybrid Answer:**
The impact of the rank 'r' hyperparameter in LoRA is that it determines the number of trainable parameters and affects the model's performance. 

As shown in Table 6, a very small rank (r=1) is suffic...

---

### Q: What is the role of the Dense Passage Retriever (DPR) in the RAG architecture?

**Metrics:**
- Baseline P@5: 0.40, NDCG@5: 0.94
- Hybrid P@5: 0.80, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines
the g...
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: In preliminary experiments, we observed that for some tasks such as story generation [ 11], the
retr...
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...

**Hybrid Retrieved Docs:**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: tasks, RAG sets a new state of the art (only on the T5-comparable split for TQA). RAG combines
the g...
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: and non-parametric memory to the “workhorse of NLP,” i.e. sequence-to-sequence (seq2seq) models.
We ...
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: pRAG-Sequence(y|x) ≈
∑
z∈top-k(p(·|x))
pη(z|x)pθ(y|x,z) =
∑
z∈top-k(p(·|x))
pη(z|x)
N∏
i
pθ(yi|x,z,y...

**Baseline Answer:**
The Dense Passage Retriever (DPR) is used as the retrieval component in the RAG architecture. It follows a bi-encoder architecture, where a dense representation of a document (d(z)) is produced by a B...

**Hybrid Answer:**
The Dense Passage Retriever (DPR) plays the role of a retriever in the RAG architecture. It is responsible for providing latent documents conditioned on the input, which are then used by the seq2seq m...

---

### Q: How does FlashAttention use tiling to reduce memory I/O compared to standard attention mechanisms?

**Metrics:**
- Baseline P@5: 0.40, NDCG@5: 0.91
- Hybrid P@5: 0.40, NDCG@5: 0.73

**Baseline Retrieved Docs:**
1. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: FlashAttention
Memory Hierarchy with
Bandwidth & Memory Size
Attention on GPT-2
FlashAttentionPyTorc...
2. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: B.5 Comparison with Rabe and Staats [66]
We describe here some similarities and diﬀerences between o...
3. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: Runtime. Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass ofFlashA...

**Hybrid Retrieved Docs:**
1. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: B.5 Comparison with Rabe and Staats [66]
We describe here some similarities and diﬀerences between o...
2. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: B.5 Comparison with Rabe and Staats [66]
We describe here some similarities and diﬀerences between o...
3. FlashAttention Fast and Memory-Efficient Exact Attention (2022).pdf: FlashAttention: Fast and Memory-Eﬃcient Exact Attention
with IO-Awareness
Tri Daoy, Daniel Y. Fuy, S...

**Baseline Answer:**
0 otherwise.
The output O =PV will be the final output of the attention mechanism.
The block-sparseFlashAttention algorithm, in the outer loop, loads a block of size𝐵𝑐 of the input matricesQ,
K, and V...

**Hybrid Answer:**
According to the provided context, FlashAttention uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. This is done to reduce memory I/...

---

### Q: How does Chain-of-Thought prompting improve symbolic reasoning compared to standard few-shot prompting?

**Metrics:**
- Baseline P@5: 0.40, NDCG@5: 0.85
- Hybrid P@5: 0.80, NDCG@5: 0.83

**Baseline Retrieved Docs:**
1. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: of the datasets only have an evaluation split, we manually composed a set of eight few-shot exemplar...
2. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: the input length of 1024 tokens), we verify that the chain-of-thought prompting is robust to various...
3. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: tion where the chain of thought prompt is only given after the
answer, isolating whether the model a...

**Hybrid Retrieved Docs:**
1. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: the input length of 1024 tokens), we verify that the chain-of-thought prompting is robust to various...
2. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: experiments on commonsense reasoning underscored how the linguistic nature of chain-of-thought
reaso...
3. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: experiments on commonsense reasoning underscored how the linguistic nature of chain-of-thought
reaso...

**Baseline Answer:**
The text does not directly compare Chain-of-Thought prompting to standard few-shot prompting for symbolic reasoning. However, it does mention that Chain-of-Thought prompting improves performance on va...

**Hybrid Answer:**
According to the provided context, Chain-of-Thought prompting facilitates OOD (Out-of-Distribution) generalization to longer sequence lengths for symbolic reasoning. This means that Chain-of-Thought p...

---

### Q: Why does LLaMA use SwiGLU activation functions instead of standard ReLU?

**Metrics:**
- Baseline P@5: 0.00, NDCG@5: 1.00
- Hybrid P@5: 0.00, NDCG@5: 0.87

**Baseline Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
r(Y...
2. LLaMA Open and Efficient Foundation Language Models (2023).pdf: place the ReLU non-linearity by the SwiGLU ac-
tivation function, introduced by Shazeer (2020) to
im...
3. LLaMA Open and Efficient Foundation Language Models (2023).pdf: to train an instruct model, LLaMA-I.
OPT 30B 26.1
GLM 120B 44.8
PaLM 62B 55.1
PaLM-cont 62B 62.8
Chi...

**Hybrid Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: place the ReLU non-linearity by the SwiGLU ac-
tivation function, introduced by Shazeer (2020) to
im...
2. LLaMA Open and Efficient Foundation Language Models (2023).pdf: place the ReLU non-linearity by the SwiGLU ac-
tivation function, introduced by Shazeer (2020) to
im...
3. LLaMA Open and Efficient Foundation Language Models (2023).pdf: params dimension nheads nlayers learning rate batch size ntokens
6.7B 4096 32 32 3.0e−4 4M 1.0T
13.0...

**Baseline Answer:**
According to the text, the reason for using SwiGLU activation functions instead of standard ReLU is that SwiGLU activation functions, introduced by Shazeer (2020), can "improve the performance."...

**Hybrid Answer:**
According to the text, LLaMA uses SwiGLU activation functions instead of standard ReLU because it was introduced by Shazeer (2020) to improve performance, and it was found to be effective in PaLM....

---

### Q: Describe the 'U-shaped' performance curve observed in long-context language models.

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 1.00
- Hybrid P@5: 0.40, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. Lost in the Middle How Language Models Use Long Contexts (2023).pdf: input question) within the language model’s input con-
text results in a U-shaped performance curve—...
2. Lost in the Middle How Language Models Use Long Contexts (2023).pdf: which found that non-instruction fine-tuned lan-
guage models are biased towards recent tokens (i.e....
3. Language Models are Few-Shot Learners.pdf: This work suggests that this trend largely continues as models continue to scale up (although a slig...

**Hybrid Retrieved Docs:**
1. Lost in the Middle How Language Models Use Long Contexts (2023).pdf: models struggle to simply retrieve matching tokens
that occur in the middle of their input context a...
2. Lost in the Middle How Language Models Use Long Contexts (2023).pdf: input question) within the language model’s input con-
text results in a U-shaped performance curve—...
3. Lost in the Middle How Language Models Use Long Contexts (2023).pdf: which found that non-instruction fine-tuned lan-
guage models are biased towards recent tokens (i.e....

**Baseline Answer:**
The 'U-shaped' performance curve observed in long-context language models is a phenomenon where models perform better at using relevant information that occurs at the very beginning (primacy bias) or ...

**Hybrid Answer:**
The 'U-shaped' performance curve observed in long-context language models refers to a phenomenon where the model's performance degrades significantly when it must access and use information located in...

---

### Q: What are the three stages of RLHF used in InstructGPT?

**Metrics:**
- Baseline P@5: 0.00, NDCG@5: 0.84
- Hybrid P@5: 0.20, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: models perform slightly worse than our SFT baseline, and labelers signiﬁcantly prefer InstructGPT
to...
2. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: policy.
Our initial RLHF experiments showed regressions on public NLP datasets, such as SQuADv2 and
...
3. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: Model&Method # Trainable WikiSQL MNLI-m SAMSum
Parameters Acc. (%) Acc. (%) R1/R2/RL
GPT-3 (FT) 175,...

**Hybrid Retrieved Docs:**
1. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: To train the very ﬁrst InstructGPT models, we asked labelers to write prompts themselves. This is
be...
2. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: models perform slightly worse than our SFT baseline, and labelers signiﬁcantly prefer InstructGPT
to...
3. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: policy.
Our initial RLHF experiments showed regressions on public NLP datasets, such as SQuADv2 and
...

**Baseline Answer:**
The text does not explicitly mention the three stages of RLHF used in InstructGPT. However, it can be inferred from the context that the RLHF training involves the following stages:

1. **Initializati...

**Hybrid Answer:**
The three stages of RLHF (Reinforcement Learning from Human Feedback) used in InstructGPT are not explicitly mentioned in the provided context. However, based on the information given, we can infer th...

---

### Q: Why are sine and cosine functions used for positional encodings in the Transformer architecture?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 1.00
- Hybrid P@5: 0.40, NDCG@5: 0.94

**Baseline Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: P E(pos,2i) = sin(pos/100002i/dmodel )
P E(pos,2i+1) = cos(pos/100002i/dmodel )
where pos is the pos...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: ture is a multi-layer bidirectional Transformer en-
coder based on the original implementation de-
s...
3. Attention Is All You Need (2017).pdf: Figure 1: The Transformer - model architecture.
The Transformer follows this overall architecture us...

**Hybrid Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: P E(pos,2i) = sin(pos/100002i/dmodel )
P E(pos,2i+1) = cos(pos/100002i/dmodel )
where pos is the pos...
2. Attention Is All You Need (2017).pdf: Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations
for ...
3. Attention Is All You Need (2017).pdf: Figure 1: The Transformer - model architecture.
The Transformer follows this overall architecture us...

**Baseline Answer:**
The sine and cosine functions are used for positional encodings in the Transformer architecture because they allow the model to easily learn to attend by relative positions. For any fixed offset k, th...

**Hybrid Answer:**
The sine and cosine functions are used for positional encodings in the Transformer architecture because it allows the model to easily learn to attend by relative positions. For any fixed offset k, the...

---

### Q: Explain the 'Masked Language Model' (MLM) pre-training objective used in BERT.

**Metrics:**
- Baseline P@5: 0.80, NDCG@5: 1.00
- Hybrid P@5: 0.80, NDCG@5: 0.97

**Baseline Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT alleviates the previously mentioned unidi-
rectionality constraint by using a “masked lan-
guag...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as Ti ∈RH.
For a given token, its input representation is
constructed by summing the corresponding t...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT including:
– Effect of Number of Training Steps; and
– Ablation for Different Masking Proce-
du...

**Hybrid Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT alleviates the previously mentioned unidi-
rectionality constraint by using a “masked lan-
guag...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT alleviates the previously mentioned unidi-
rectionality constraint by using a “masked lan-
guag...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: GPT) (Radford et al., 2018), introduces minimal
task-speciﬁc parameters, and is trained on the
downs...

**Baseline Answer:**
The 'Masked Language Model' (MLM) pre-training objective used in BERT is inspired by the Cloze task. In this objective, a "masked language model" randomly masks some of the tokens from the input, and ...

**Hybrid Answer:**
The 'Masked Language Model' (MLM) pre-training objective used in BERT is a technique inspired by the Cloze task (Taylor, 1953). In this approach, the model randomly masks some of the tokens from the i...

---

### Q: How does LoRA allow for zero additional inference latency when deployed?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 0.97
- Hybrid P@5: 0.60, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary
r(Y...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: usage by up to 2/3 if r ≪dmodel as we do not need to store the optimizer states for the frozen
param...
3. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: greater efﬁciency (AdapterD). We cite numbers from prior works whenever possible to maximize
the num...

**Hybrid Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: LoRA, an efﬁcient adaptation strategy that neither introduces inference latency nor reduces input
se...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: LoRA, an efﬁcient adaptation strategy that neither introduces inference latency nor reduces input
se...
3. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: ing fewer trainable parameters, a higher training throughput, and, unlike adapters,
no additional in...

**Baseline Answer:**
When deployed, LoRA allows for zero additional inference latency by explicitly computing and storing W = W0 + BA, where W0 is the pre-trained weight matrix, B is the LoRA bias matrix, and A is the LoR...

**Hybrid Answer:**
LoRA allows for zero additional inference latency when deployed by either absorbing the LoRA weights (A and B) into the original model weights (W) or by not merging the weights and dynamically choosin...

---

### Q: Which paper first demonstrated emergent abilities in large language models, and what does 'emergent' mean in this context?

**Metrics:**
- Baseline P@5: 0.00, NDCG@5: 1.00
- Hybrid P@5: 0.20, NDCG@5: 0.98

**Baseline Retrieved Docs:**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: arXiv:1704.05179 [cs], April 2017. URL http://arxiv.org/abs/1704.05179. arXiv:
1704.05179.
[11] Ange...
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: [7] Christopher Clark and Matt Gardner. Simple and Effective Multi-Paragraph Reading Compre-
hension...
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020).pdf: from Question-Answer Pairs. In Proceedings of the 2013 Conference on Empirical Methods
in Natural La...

**Hybrid Retrieved Docs:**
1. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Ma...
2. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: errors in all categories.
There are also three notable points regarding why small language models fa...
3. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: A Frequently Asked Questions
A.1 Why does increasing model scale improve chain-of-thought prompting?...

**Baseline Answer:**
Based on the context provided, I couldn't find any specific paper mentioned that first demonstrated emergent abilities in large language models. However, I can provide some general information about e...

**Hybrid Answer:**
The paper that first demonstrated emergent abilities in large language models is:

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denn...

---

### Q: How did the Transformer architecture eliminate recurrence when modeling long-range dependencies?

**Metrics:**
- Baseline P@5: 0.60, NDCG@5: 0.94
- Hybrid P@5: 1.00, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: See Section F.2 for more details on the plotted data points.
6 R ELATED WORKS
Transformer Language M...
2. LLaMA Open and Efficient Foundation Language Models (2023).pdf: In the rest of this paper, we present an overview
of the modiﬁcations we made to the transformer
arc...
3. Attention Is All You Need (2017).pdf: Attention mechanisms have become an integral part of compelling sequence modeling and transduc-
tion...

**Hybrid Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: Attention mechanisms have become an integral part of compelling sequence modeling and transduc-
tion...
2. Attention Is All You Need (2017).pdf: Attention mechanisms have become an integral part of compelling sequence modeling and transduc-
tion...
3. Attention Is All You Need (2017).pdf: in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
...

**Baseline Answer:**
The Transformer architecture eliminated recurrence by relying entirely on an attention mechanism to draw global dependencies between input and output, rather than using a recurrent network. This allow...

**Hybrid Answer:**
The Transformer architecture eliminated recurrence when modeling long-range dependencies by relying entirely on an attention mechanism to draw global dependencies between input and output sequences, r...

---

### Q: Why is LLaMA particularly important for enterprise and on-premise deployment of large language models?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 1.00
- Hybrid P@5: 0.20, NDCG@5: 0.81

**Baseline Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: LLaMA: Open and Efﬁcient Foundation Language Models
Hugo Touvron∗, Thibaut Lavril∗, Gautier Izacard ...
2. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: 2014).
However, making language models better at following user intentions also makes them easier to...
3. LLaMA Open and Efficient Foundation Language Models (2023).pdf: improve even after 1T tokens.
The focus of this work is to train a series of
language models that ac...

**Hybrid Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: improve even after 1T tokens.
The focus of this work is to train a series of
language models that ac...
2. LLaMA Open and Efficient Foundation Language Models (2023).pdf: LLaMA: Open and Efﬁcient Foundation Language Models
Hugo Touvron∗, Thibaut Lavril∗, Gautier Izacard ...
3. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: 2014).
However, making language models better at following user intentions also makes them easier to...

**Baseline Answer:**
The text does not explicitly state why LLaMA is particularly important for enterprise and on-premise deployment of large language models. However, it mentions several points that can be inferred as be...

**Hybrid Answer:**
The provided text does not explicitly mention why LLaMA is particularly important for enterprise and on-premise deployment of large language models. However, based on the context, it can be inferred t...

---

### Q: Compare the bidirectional attention mechanism in BERT with the unidirectional attention in GPT-style models.

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 0.71
- Hybrid P@5: 0.00, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: the majority of the empirical improvements, but
we do note that there are several other differences
...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as natural language inference (Conneau et al.,
2017) and machine translation (McCann et al.,
2017). ...
3. Language Models are Few-Shot Learners.pdf: This could also explain the comparatively low scores for RTE and CB, which also follow this format. ...

**Hybrid Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: extract long contiguous sequences.
3.2 Fine-tuning BERT
Fine-tuning is straightforward since the sel...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: extract long contiguous sequences.
3.2 Fine-tuning BERT
Fine-tuning is straightforward since the sel...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: els, as ELMo does. However: (a) this is twice as
expensive as a single bidirectional model; (b) this...

**Baseline Answer:**
The bidirectional attention mechanism in BERT is compared to the unidirectional attention in GPT-style models as follows:

BERT uses a multi-layer bidirectional Transformer encoder. This means that th...

**Hybrid Answer:**
The bidirectional attention mechanism in BERT is compared to the unidirectional attention in GPT-style models in the following points:

- BERT uses a self-attention mechanism to unify the encoding of ...

---

### Q: Compare the 'Few-Shot Prompting' approach proposed in GPT-3 with the 'RLHF' approach introduced in InstructGPT regarding model weight updates.

**Metrics:**
- Baseline P@5: 0.00, NDCG@5: 1.00
- Hybrid P@5: 0.00, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: 0.25
0.50
0.75Win rate against SFT 175B
GPT distribution
GPT GPT
(prompted) SFT PPO PPO-ptx
Instruct...
2. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: models perform slightly worse than our SFT baseline, and labelers signiﬁcantly prefer InstructGPT
to...
3. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: Figure 2: A diagram illustrating the three steps of our method: (1) supervised ﬁne-tuning (SFT), (2)...

**Hybrid Retrieved Docs:**
1. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: models perform slightly worse than our SFT baseline, and labelers signiﬁcantly prefer InstructGPT
to...
2. Training Language Models to Follow Instructions (InstructGPT) (2022).pdf: Figure 2: A diagram illustrating the three steps of our method: (1) supervised ﬁne-tuning (SFT), (2)...
3. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: task-speciﬁc ﬁnetuned models on several tasks, even achieving new state of the art on the challengin...

**Baseline Answer:**
The 'Few-Shot Prompting' approach proposed in GPT-3 and the 'RLHF' (Reward Learning from Human Feedback) approach introduced in InstructGPT differ in terms of model weight updates.

In the 'Few-Shot P...

**Hybrid Answer:**
The 'Few-Shot Prompting' approach proposed in GPT-3 and the 'RLHF' approach introduced in InstructGPT differ in their model weight update strategies.

The 'Few-Shot Prompting' approach in GPT-3 involv...

---

### Q: What is the specific parameter count of the largest model presented in the LLaMA (2023) paper?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 0.96
- Hybrid P@5: 0.40, NDCG@5: 0.95

**Baseline Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: to train an instruct model, LLaMA-I.
OPT 30B 26.1
GLM 120B 44.8
PaLM 62B 55.1
PaLM-cont 62B 62.8
Chi...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: greater efﬁciency (AdapterD). We cite numbers from prior works whenever possible to maximize
the num...
3. LLaMA Open and Efficient Foundation Language Models (2023).pdf: 7B 13B 33B 65B
All 66.0 64.7 69.0 77.5
her/her/she 65.0 66.7 66.7 78.8
his/him/he 60.8 62.5 62.1 72....

**Hybrid Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: improve even after 1T tokens.
The focus of this work is to train a series of
language models that ac...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: For example, the largest Transformer explored in
Vaswani et al. (2017) is (L=6, H=1024, A=16)
with 1...
3. LLaMA Open and Efficient Foundation Language Models (2023).pdf: to train an instruct model, LLaMA-I.
OPT 30B 26.1
GLM 120B 44.8
PaLM 62B 55.1
PaLM-cont 62B 62.8
Chi...

**Baseline Answer:**
The specific parameter count of the largest model presented in the LLaMA (2023) paper is not directly mentioned in the provided snippet. However, based on the text, we can infer that the largest model...

**Hybrid Answer:**
The specific parameter count of the largest model presented in the LLaMA (2023) paper is 65B parameters....

---

### Q: What was the exact percentage improvement in BLEU score reported in the original Transformer paper?

**Metrics:**
- Baseline P@5: 0.60, NDCG@5: 0.93
- Hybrid P@5: 1.00, NDCG@5: 0.94

**Baseline Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
En...
2. Attention Is All You Need (2017).pdf: Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the b...
3. Language Models are Few-Shot Learners.pdf: Ro→En 16 test bleu-sb 12 41.3 1999 40.4 423 41.6 1576 79% 1%
WebQs test acc 8 41.5 2032 41.6 428 41....

**Hybrid Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
En...
2. Attention Is All You Need (2017).pdf: 6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transf...
3. Attention Is All You Need (2017).pdf: 6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transf...

**Baseline Answer:**
The original Transformer paper achieved a new state-of-the-art BLEU score of 28.4 on the English-to-German translation task. The previous best model achieved a BLEU score of 26.30. To find the exact p...

**Hybrid Answer:**
The original Transformer paper does not explicitly mention the exact percentage improvement in BLEU score. However, it does mention that the big transformer model outperforms the best previously repor...

---

### Q: Does the BERT architecture include a decoder stack during pre-training?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 1.00
- Hybrid P@5: 0.00, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT BERT
E[CLS] E1  E[SEP]... EN E1’ ... EM’
C
 T1
 T[SEP]...
 TN
 T1’ ...
 TM’
[CLS] Tok 1  [SEP]....
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as Ti ∈RH.
For a given token, its input representation is
constructed by summing the corresponding t...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as natural language inference (Conneau et al.,
2017) and machine translation (McCann et al.,
2017). ...

**Hybrid Retrieved Docs:**
1. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as Ti ∈RH.
For a given token, its input representation is
constructed by summing the corresponding t...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: as natural language inference (Conneau et al.,
2017) and machine translation (McCann et al.,
2017). ...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding
Jacob Devlin Ming-W...

**Baseline Answer:**
No, the BERT architecture does not include a decoder stack during pre-training. According to the text, BERT uses a multi-layer bidirectional Transformer encoder, which is a "Transformer encoder" while...

**Hybrid Answer:**
No, the BERT architecture does not include a decoder stack during pre-training. According to the text, the BERT model architecture is a multi-layer bidirectional Transformer encoder, which is based on...

---

### Q: According to the original Transformer paper, what happens to performance when the number of attention heads is reduced?

**Metrics:**
- Baseline P@5: 0.20, NDCG@5: 0.96
- Hybrid P@5: 0.20, NDCG@5: 0.75

**Baseline Retrieved Docs:**
1. Attention Is All You Need (2017).pdf: output values. These are concatenated and once again projected, resulting in the final values, as
de...
2. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: guarantees that we do not introduce any additional latency during inference compared to a ﬁne-tuned
...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: ture is a multi-layer bidirectional Transformer en-
coder based on the original implementation de-
s...

**Hybrid Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: guarantees that we do not introduce any additional latency during inference compared to a ﬁne-tuned
...
2. BERT Pre-training of Deep Bidirectional Transformers for.pdf: ture is a multi-layer bidirectional Transformer en-
coder based on the original implementation de-
s...
3. BERT Pre-training of Deep Bidirectional Transformers for.pdf: ture is a multi-layer bidirectional Transformer en-
coder based on the original implementation de-
s...

**Baseline Answer:**
The original Transformer paper (Vaswani et al., 2017) does not explicitly state what happens to performance when the number of attention heads is reduced. However, the text provided does mention the f...

**Hybrid Answer:**
The original text does not directly answer the question about the performance when the number of attention heads is reduced according to the original Transformer paper. However, it does mention that w...

---

### Q: In which season do we get mangoes?

**Metrics:**
- Baseline P@5: 0.00, NDCG@5: 0.86
- Hybrid P@5: 0.00, NDCG@5: 1.00

**Baseline Retrieved Docs:**
1. LLaMA Open and Efficient Foundation Language Models (2023).pdf: provided context. We follow Gao et al. (2021)
and use the likelihood normalized by the number
of cha...
2. LLaMA Open and Efficient Foundation Language Models (2023).pdf: BoolQ PIQA SIQA HellaSwag WinoGrande ARC-e ARC-c OBQA
GPT-3 175B 60.5 81.0 - 78.9 70.2 68.8 51.4 57....
3. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more tha...

**Hybrid Retrieved Docs:**
1. LoRA Low-Rank Adaptation of Large Language Models (2021).pdf: Method Dataset MNLI SST-2 MRPC CoLA QNLI QQP RTE STS-B
Optimizer AdamW
Warmup Ratio 0.06
LR Schedule...
2. Chain-of-Thought Prompting Elicits Reasoning (2022).pdf: A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more tha...
3. Language Models are Few-Shot Learners.pdf: got the ball, he was tackled for a one-yard loss while attempting to throw
his first NFL pass on a h...

**Baseline Answer:**
Based on the given context, there is no direct information about the question "In which season do we get mangoes?" and its corresponding answer. However, the given information is related to various na...

**Hybrid Answer:**
The answer is not directly available in the given context. However, based on general knowledge, mangoes are typically available during the summer season in many parts of the world. In the Northern Hem...

---

