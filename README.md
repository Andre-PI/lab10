# Laboratório 10 — O Pipeline Definitivo
## RAG, QLoRA e Otimização de Inferência na GPU

> **Declaração:** Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **Andre Lucas Francino Castelo Branco**.
> - codigo foi gerador por IA
> - decisões do desenvolvimento como a mudança pra sdpa e redução de tokens foi feita por André Lucas
> - otimização do texto deste readme foi gerado por IA
> - texto pra rag feito por IA
> - 
---

## Ambiente de Execução

| Item | Valor |
|---|---|
| Plataforma | Google Colab — GPU Tesla T4 (14,6 GB VRAM) |
| Modelo | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Quantização | NF4 4-bit via bitsandbytes |
| Framework | PyTorch 2.x + HuggingFace Transformers 4.41+ |
| Atenção eficiente | SDPA nativo do PyTorch 2.x |

---

## Métricas de Benchmark

### Passo 1 — Carga do Modelo (QLoRA 4-bit)

| Métrica | Valor |
|---|---|
| VRAM ocupada após carga do modelo em 4-bit | **727,2 MB** |
| VRAM estimada sem quantização (Float16) | ~2.200 MB |
| Redução pela quantização | **~3×** |

### Passo 3 vs Passo 4 — Comparativo de Geração

| Métrica | Sem Otimização (P3) | Com KV Cache + SDPA (P4) |
|---|---|---|
| Contexto (tokens) | 4.000 | 5.000 |
| KV Cache | Não | Sim |
| Atenção eficiente (SDPA/FA2) | Não | Sim |
| Tempo de geração (s) | **380,80** | **23,01** |
| Tokens por segundo | **0,26** | **4,35** |
| Pico de VRAM (MB) | **5.619,8** | **8.050,3** |
| Speedup | — | **16,6×** |

---

## Observações Importantes sobre a Execução

### Por que usamos 4.000–5.000 tokens em vez dos 10.000–15.000 do enunciado?

O enunciado pede um contexto de 10.000 a 15.000 tokens, mas durante os testes descobrimos um limitador arquitetural importante: o modelo TinyLlama-1.1B tem uma janela de contexto máxima de **2.048 tokens**, definida no seu arquivo de configuração (`max_position_embeddings: 2048`). Qualquer entrada acima disso faz o modelo entrar em território não treinado, gerando saídas sem sentido — o que pode ser observado na última célula do notebook, onde o texto gerado é repetitivo ("e e e e..."), exatamente por causa disso.

Além disso, quando tentamos rodar o Passo 4 com 12.000 tokens, a GPU entrou em **Out-Of-Memory (OOM)** mesmo com as otimizações ativas — o erro foi:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 17.17 GiB.
```
Esse crash é na prática **a demonstração mais concreta do problema descrito no enunciado**: a complexidade O(n²) da atenção padrão tornando inviável processar contextos grandes. Optamos por reduzir para 4.000–5.000 tokens para conseguir completar o benchmark comparativo (Passo 3 vs Passo 4), que é o objetivo central do laboratório.

### Por que usamos `attn_implementation='sdpa'` em vez de `'flash_attention_2'`?

O pacote `flash-attn` precisa ser compilado do zero e falhou no ambiente Colab com erro de build. O `sdpa` (Scaled Dot Product Attention) é a implementação **nativa do PyTorch 2.x** que usa o mesmo algoritmo de tiles em memória SRAM que o FlashAttention-2, com a mesma complexidade O(n) de memória. Para os fins deste laboratório, os dois demonstram o mesmo princípio arquitetural.

---

## Análise Arquitetural

### Parte A — Como QLoRA, KV Cache e FlashAttention salvaram o Transformer do colapso de VRAM

A primeira coisa que fizemos foi carregar o modelo com quantização QLoRA em 4-bit. Sem isso, o TinyLlama ocuparia cerca de 2.200 MB de VRAM só com os pesos — com a quantização NF4, esse número caiu para 727 MB, liberando espaço essencial para o restante do pipeline. A ideia por trás disso é simples: em vez de guardar cada número do modelo com 16 bits de precisão, guardamos com 4 bits, e só na hora de fazer as contas convertemos de volta para 16 bits. Com a VRAM dos pesos liberada, o próximo problema aparece na fase de geração: sem o KV Cache, o modelo recalcula do zero os vetores de atenção (Q, K e V) para todos os tokens anteriores a cada novo token gerado. Com 4.000 tokens de contexto, isso significa que gerar 100 tokens exigiu 100 passagens completas pelo modelo — o que nos deu 380 segundos para uma tarefa que deveria levar segundos. O KV Cache resolve isso guardando os vetores K e V já calculados e reaproveitando-os nos passos seguintes, o que levou o tempo para 23 segundos — 16,6 vezes mais rápido. Por fim, o FlashAttention (aqui implementado via SDPA do PyTorch) cuida do problema de memória da própria operação de atenção: em vez de montar a matriz inteira de scores de atenção na memória global da GPU (o que cresce com O(n²) conforme o contexto aumenta), ele processa a atenção em blocos menores usando a memória on-chip da GPU, mais rápida e limitada. Foram essas três técnicas juntas que permitiram o pipeline funcionar.

### Parte B — Por que o FlashAttention falharia com 2 milhões de tokens e por que precisaríamos do Mamba

Mesmo com FlashAttention resolvendo o crescimento quadrático da atenção, existe outro problema que ele não consegue resolver: o KV Cache cresce de forma linear com o tamanho do contexto. Cada camada do modelo precisa guardar os vetores K e V de todos os tokens que já foram processados. Para um contexto de 2 milhões de tokens no TinyLlama (22 camadas, 4 heads KV, dimensão 64, Float16), só o KV Cache exigiria aproximadamente 89 GB de memória — o que é inviável em qualquer GPU disponível hoje, seja para uso pessoal ou em produção. É por isso que a indústria começou a olhar para arquiteturas diferentes, como o **Mamba**, que pertence à família dos State Space Models (SSMs). A diferença fundamental é que o Mamba não precisa guardar todos os tokens anteriores: ele mantém um **estado interno de tamanho fixo** que vai sendo atualizado a cada novo token, como se fosse uma "memória comprimida" do que já foi processado. Isso faz com que o consumo de memória seja O(1) — ou seja, constante, independente do tamanho do contexto. O preço disso é que o Mamba pode perder informações específicas de tokens muito distantes no passado, o que o torna menos preciso do que a atenção para tarefas que exigem recuperar detalhes exatos. Mas para contextos com milhões de tokens — como análise de documentos inteiros, genomas ou séries temporais longas — o Mamba é hoje a solução arquitetural mais viável.

---

## Dependências

```
transformers>=4.41.0
bitsandbytes>=0.43.1
accelerate>=0.30.0
torch>=2.1.0
```
