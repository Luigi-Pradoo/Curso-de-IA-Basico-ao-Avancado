# Aula 5.2 — Treino, validação e teste

> **Módulo 05 · Fundamentos práticos do ML** · 2 h · Dataset: **California Housing** (regressão)
> Material: [`Aula 5.2.pdf`](Aula%205.2.pdf) (33 slides)

Na Aula 5.1 a pergunta era *qual é a métrica?*. Hoje ela muda para **a métrica é honesta?**. Aparece um terceiro conjunto — a **validação** — e o conceito mais importante do módulo: **vazamento de dados**, qualquer informação do teste que chega ao treino. O código roda sem erro, o número sobe, o relatório sai bonito… e em produção o modelo volta ao nível de antes.

---

## Objetivos

1. Explicar o papel de treino, validação e teste.
2. Justificar por que o teste fica **trancado**.
3. Definir vazamento de dados com um exemplo.
4. Identificar um vazamento num notebook alheio.
5. Corrigir a ordem das etapas e medir de novo.
6. Ler R², MAE e RMSE com desconfiança saudável.

---

## Teoria

### Um dataset, três papéis

| Conjunto | Papel |
|---|---|
| **Treino** | o modelo ajusta os próprios parâmetros |
| **Validação** | você escolhe o modelo e os hiperparâmetros (pode consultar quantas vezes quiser) |
| **Teste** | estima o desempenho final — **uma única vez** |

> Se você usa o teste para tomar decisões, ele deixa de ser um teste independente: vira mais um conjunto de validação, e você fica sem nenhuma estimativa limpa do mundo real.

**O conjunto trancado:** ~20% das linhas, separadas na primeira célula. Não olhar, não ajustar, não escolher modelo com ele. Abrir só no final, com o modelo já escolhido.

### A regra de ouro
> **Toda transformação que aprende algo dos dados deve aprender somente com o treino.**

`StandardScaler`, `SimpleImputer`, `OneHotEncoder` e seleção de features guardam números tirados do dado que veem. Se virem o teste, o teste já vazou.

| ❌ Com vazamento | ✅ Sem vazamento |
|---|---|
| dados → **escalonar** → split → treino/teste | dados → **split** → escalonar (aprende só no treino) → teste passa pela escala pronta |

### Cinco formas de vazar
1. **Pré-processamento antes do split** — a estatística da transformação já viu o teste.
2. **Linhas duplicadas** — a mesma amostra nos dois conjuntos.
3. **Feature com informação futura** — a coluna só existe depois do fato que se quer prever.
4. **Dados temporais misturados** — treinar com amanhã para prever ontem.
5. **Mesmo grupo nos dois lados** — várias linhas do mesmo cliente, paciente ou sensor.

### E quando sobra pouco dado?
A validação vira um giro (**validação cruzada**): cada parte do treino é validação uma vez. O teste continua trancado. (Implementada do zero na [Aula 6.6](../../Aula6/Aula%206.6/README.md).)

---

## Exercício guiado — uma investigação em 5 etapas

Um notebook "chegou pronto": split existe, semente fixa, código roda sem aviso, e o **R² no teste é 0,99** prevendo preço de imóvel.

| # | Etapa | O que acontece |
|---|---|---|
| 1 | Um modelo aparentemente excelente | R² = 0,991 — plausível para preço de casa? |
| 2 | Investigação: 4 perguntas | transformação antes do split? duplicatas? feature quase igual ao alvo? coluna que só existe depois do fato? |
| 3 | Descoberta | `valor_estimado` tem correlação **0,998** com o alvo — é calculada a partir do próprio preço |
| 4 | Correção | tirar a coluna vazada **e** escalonar depois do split (`make_pipeline(StandardScaler(), Ridge())`) |
| 5 | Comparação honesta | ver tabela abaixo |

| Métrica | Com vazamento | Correto |
|---|---|---|
| R² | 0,991 | **0,576** |
| MAE | 0,04 | **0,53** (≈ US$ 53 mil de erro médio) |
| RMSE | 0,11 | **0,75** |

> **Uma métrica maior nem sempre significa um modelo melhor. A primeira pergunta é: a avaliação é honesta?**
> Métrica quase perfeita é **hipótese de vazamento**, não prova de talento.

---

## Erros comuns

| Erro | Como evitar |
|---|---|
| Pré-processar antes do split | o split é a primeira linha depois de carregar |
| Ignorar duplicatas | contar duplicatas **antes** de separar |
| Usar informação futura | perguntar de cada coluna: *eu teria isso no momento da previsão?* |
| Embaralhar dado temporal | com tempo no dado, treino é passado e teste é futuro |
| Separar grupos relacionados | o grupo inteiro fica de um lado só do split |
| Escolher o modelo pelo teste | decidir na validação; abrir o teste uma vez, no fim |

## Desafio de casa

Entregue **duas versões** do mesmo notebook (mesma semente): uma com um vazamento **proposital** e outra corrigida. Dataset livre (California Housing, `load_diabetes` ou um dado seu).
Explique em texto: onde ocorreu o vazamento, por que aconteceu, como foi corrigido e como R², MAE e RMSE mudaram.
**Critério de aceite:** apontar exatamente a linha onde o vazamento ocorreu e justificar a correção.

## Leitura
- scikit-learn — *Common pitfalls: data leakage*
- `train_test_split`
- *Cross-validation: evaluating estimator performance*
- Métricas de regressão: R², MAE e RMSE

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 5.2.pdf` | slides da aula |

---

**← Anterior:** [Aula 5.1 — O pipeline completo](../Aula%205.1/README.md) · **Próxima →** [Aula 5.3 — Pipeline e ColumnTransformer](../Aula%205.3/README.md)
