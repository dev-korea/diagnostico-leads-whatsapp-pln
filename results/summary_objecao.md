| method                           | model_version                         |   n_test | F1-macro [IC95%]                         |
|:---------------------------------|:--------------------------------------|---------:|:-----------------------------------------|
| bertimbau                        | neuralmind/bert-base-portuguese-cased |       33 | 0.139 [0.075; 0.222] ±0.064 (5 sementes) |
| llm_fewshot_k3:ollama-qwen2.5-7b | qwen2.5:7b@845dbda0ea48               |       33 | 0.638 [0.311; 0.818]                     |
| tfidf_lr                         | C=0.1                                 |       33 | 0.315 [0.095; 0.485]                     |

Comparações pareadas (mesmo teste):

| A                                | B                                | Δ F1 (A−B) [IC95%]      |   mcnemar_p |   mcnemar_p_holm |
|:---------------------------------|:---------------------------------|:------------------------|------------:|-----------------:|
| bertimbau                        | llm_fewshot_k3:ollama-qwen2.5-7b | -0.400 [-0.555; -0.172] |      0.0000 |           0.0000 |
| bertimbau                        | tfidf_lr                         | -0.077 [-0.290; +0.154] |      0.0000 |           0.0000 |
| llm_fewshot_k3:ollama-qwen2.5-7b | tfidf_lr                         | +0.323 [-0.020; +0.606] |      0.0312 |           0.0312 |