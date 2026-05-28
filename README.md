# Laboratório 10 — O Pipeline Definitivo
## RAG, QLoRA e Otimização de Inferência na GPU

> **Declaração:** Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **Andre Lucas Francino Castelo Branco**.

---

## Ambiente de Execução

| Item | Valor |
|---|---|
| Plataforma | Google Colab — GPU T4 (15 GB VRAM) |
| Modelo | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Quantização | NF4 4-bit (bitsandbytes) |
| Framework | PyTorch 2.x + HuggingFace Transformers 4.41+ |

---

## Métricas de Benchmark

### Passo 1 — Carga do Modelo (QLoRA 4-bit)

| Métrica | Valor |
|---|---|
| VRAM ocupada após carga do modelo | **preencher após execução** MB |

> Referência: o mesmo modelo em Float16 ocuparia ~2,2 GB. A quantização 4-bit reduz para ~700 MB.

### Passo 3 vs Passo 4 — Comparativo de Geração

| Métrica | Sem Otimização (P3) | Com KV Cache + FA2 (P4) |
|---|---|---|
| Contexto (tokens) | 4.000 | 12.000 |
| KV Cache | Não | Sim |
| FlashAttention-2 | Não | Sim |
| Tempo de geração (s) | **preencher** | **preencher** |
| Tokens por segundo | **preencher** | **preencher** |
| Pico de VRAM (MB) | **preencher** | **preencher** |
| Speedup | — | **preencher**× |

> Substitua os campos **preencher** com os valores impressos pelo notebook ao final da execução.

---

## Como Executar

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do arquivo `lab10_pipeline_definitivo.ipynb`
3. Vá em **Runtime → Change runtime type → T4 GPU**
4. Execute **Runtime → Run all** e aguarde
5. Copie as métricas impressas para a tabela acima

---

## Análise Arquitetural

### Parte A — Como QLoRA, KV Cache e FlashAttention-2 salvaram o Transformer do colapso de VRAM

A primeira linha de defesa contra o colapso de VRAM é a quantização QLoRA em 4-bit: ao representar os pesos do modelo em NF4 (NormalFloat4) em vez de Float16, reduzimos o footprint dos parâmetros em aproximadamente 4×, liberando cerca de 1,5 GB de VRAM que de outro modo estariam bloqueados pelos pesos do modelo — sem degradação significativa de qualidade, pois os cálculos internos ainda ocorrem em Float16. Com esse espaço criado na memória, o KV Cache entra como segunda otimização crítica: em vez de recalcular os vetores de Query, Key e Value para todos os tokens anteriores a cada novo token gerado, os tensores K e V são armazenados em memória entre os passos de decodificação, reduzindo cada passo de O(n) de computação para um único forward pass sobre o novo token contra os vetores cacheados. Por fim, o FlashAttention-2 elimina o gargalo remanescente: a operação de atenção padrão materializa a matriz completa de scores QKᵀ na HBM (memória global da GPU, de acesso lento), exigindo O(n²) de memória por camada; o FA2 implementa tiling sobre a SRAM (memória on-chip, ordens de magnitude mais rápida), reduzindo o footprint da atenção de O(n²) para O(n) e aumentando drasticamente o throughput por aproveitar a hierarquia de memória do hardware. A combinação das três técnicas — QLoRA comprimindo pesos, KV Cache eliminando recálculo redundante e FlashAttention-2 tornando a atenção hardware-aware — permitiu processar um contexto de 12.000 tokens que, sem nenhuma otimização, ultrapassaria os limites de VRAM da GPU utilizada.

### Parte B — Por que FlashAttention-2 falharia com 2 milhões de tokens e por que a indústria precisa do Mamba

Ainda que o FlashAttention-2 resolva o problema da memória quadrática da operação de atenção em si, ele não elimina o crescimento linear do KV Cache: para cada camada do Transformer, é obrigatório armazenar os vetores K e V de todos os tokens do contexto. Para o TinyLlama (22 camadas, 4 heads KV, dimensão de head 64, Float16), um contexto de 2 milhões de tokens exigiria 2 × 22 × 4 × 64 × 2.000.000 × 2 bytes ≈ 89 GB apenas para o KV Cache — impossível em qualquer GPU de consumo e inviável em produção para múltiplas requisições simultâneas em hardware de datacenter. A solução arquitetural são os State Space Models (SSMs) como o Mamba: em vez de calcular atenção entre todos os pares de tokens, o Mamba mantém um estado recorrente de tamanho fixo — uma espécie de "memória comprimida" da sequência inteira — atualizado a cada token com operações matriciais seletivas. Isso garante complexidade O(1) de memória e O(n) de tempo independentemente do comprimento da sequência, tornando o processamento de milhões de tokens viável com o mesmo footprint de memória de uma sequência curta. O trade-off é real: SSMs perdem a capacidade de recuperação precisa e localizada de tokens específicos do passado distante (ponto forte do mecanismo de atenção), mas para tarefas como sumarização de documentos longos, análise de séries temporais e processamento de sequências genômicas, o Mamba representa a evolução natural dos Transformers quando o contexto cresce além das fronteiras do que qualquer variante de atenção — incluindo a FlashAttention-2 — pode suportar com eficiência de memória.

---

## Dependências

```
transformers>=4.41.0
bitsandbytes>=0.43.1
accelerate>=0.30.0
flash-attn>=2.5.0
torch>=2.1.0
```
