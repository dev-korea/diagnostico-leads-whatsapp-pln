| method                           | model_version                         |   n_test | F1-macro [IC95%]                         |
|:---------------------------------|:--------------------------------------|---------:|:-----------------------------------------|
| bertimbau                        | neuralmind/bert-base-portuguese-cased |       33 | 0.909 [0.805; 0.987] ±0.021 (5 sementes) |
| llm_fewshot_k3:ollama-qwen2.5-7b | qwen2.5:7b@845dbda0ea48               |       33 | 0.814 [0.656; 0.937]                     |
| tfidf_lr                         | C=10.0                                |       33 | 0.909 [0.793; 1.000]                     |

Comparações pareadas (mesmo teste):

| A                                | B                                | Δ F1 (A−B) [IC95%]      |   mcnemar_p |   mcnemar_p_holm |
|:---------------------------------|:---------------------------------|:------------------------|------------:|-----------------:|
| bertimbau                        | llm_fewshot_k3:ollama-qwen2.5-7b | +0.095 [-0.030; +0.244] |      0.3750 |           1.0000 |
| bertimbau                        | tfidf_lr                         | +0.000 [-0.093; +0.090] |      1.0000 |           1.0000 |
| llm_fewshot_k3:ollama-qwen2.5-7b | tfidf_lr                         | -0.095 [-0.257; +0.059] |      0.4531 |           1.0000 |