TEST_QUERIES = [
    # These test if the model can catch hyphenated terms and specific variable names.
    "What is the difference between RAG-Sequence and RAG-Token?",
    "What is the impact of the rank 'r' hyperparameter in LoRA?",
    "What is the role of the Dense Passage Retriever (DPR) in the RAG architecture?",

    # These queries use words like "Attention" or "Prompting" that appear in ALL papers.
    "How does FlashAttention use tiling to reduce memory I/O compared to standard attention mechanisms?",
    "How does Chain-of-Thought prompting improve symbolic reasoning compared to standard few-shot prompting?",
    "Why does LLaMA use SwiGLU activation functions instead of standard ReLU?",

    # These ask for specific lists, curves, or definitions buried deep in the text.
    "Describe the 'U-shaped' performance curve observed in long-context language models.",
    "What are the three stages of RLHF used in InstructGPT?",
    "Why are sine and cosine functions used for positional encodings in the Transformer architecture?",
    "Explain the 'Masked Language Model' (MLM) pre-training objective used in BERT.",
    "How does LoRA allow for zero additional inference latency when deployed?",

    # These test if the model can summarize big ideas without getting lost in details.
    "Which paper first demonstrated emergent abilities in large language models, and what does 'emergent' mean in this context?",
    "How did the Transformer architecture eliminate recurrence when modeling long-range dependencies?",
    "Why is LLaMA particularly important for enterprise and on-premise deployment of large language models?",

    # These require information from multiple different papers or sections to answer fully.
    "Compare the bidirectional attention mechanism in BERT with the unidirectional attention in GPT-style models.",
    "Compare the 'Few-Shot Prompting' approach proposed in GPT-3 with the 'RLHF' approach introduced in InstructGPT regarding model weight updates.",

    # Vector search hates numbers. BM25 loves them.
    "What is the specific parameter count of the largest model presented in the LLaMA (2023) paper?",
    "What was the exact percentage improvement in BLEU score reported in the original Transformer paper?",

    # Testing if the model knows what is NOT there or what happens when things break.
    "Does the BERT architecture include a decoder stack during pre-training?",
    "According to the original Transformer paper, what happens to performance when the number of attention heads is reduced?",

    # EXPECTED RESULT: Score 1 (Irrelevant). Precision should be 0.0.
    # If the system tries to match "season" to "session" or "reasoning", the Judge should punish it.
    "In which season do we get mangoes?"
]