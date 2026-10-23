# Aula 6.5 — Métricas implementadas na mão

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Exemplos: filtro de spam (binário) e pinguins (3 classes)
> Material: [`Aula 6.5 - Metricas.pdf`](Aula%206.5%20-%20Metricas.pdf) (39 slides) + [`Exercicio Guiado - Aula 6.5.py`](Exercicio%20Guiado%20-%20Aula%206.5.py)

A regressão logística da 6.4 devolve probabilidades e, com um limiar, classes. Agora a pergunta é **como saber se ela classifica bem**. "90 acertos em 100" pode ser ótimo ou péssimo — **depende** de quais foram os 10 erros (mandar um doente para casa custa mais do que pedir um exame extra) e de quantos exemplos há em cada classe (se 90 de 100 mensagens são normais, responder "normal" sempre também acerta 90). Nesta aula toda métrica nasce de **contagens**, antes de qualquer chamada ao scikit-learn.

---

## Objetivos

1. Montar a matriz de confusão a partir de `y_real` e `y_pred` e nomear TP, TN, FP e FN.
2. Calcular accuracy, precision, recall e F1 à mão e explicar o que cada uma mede.
3. Escolher a métrica adequada a partir do **custo de cada tipo de erro**.
4. Construir a curva ROC variando o limiar e calcular a AUC pela regra dos trapézios.
5. Escrever o seu próprio `classification_report` em NumPy e conferir com o scikit-learn.

---

## Teoria

### Matriz de confusão (convenção: linhas = real, colunas = previsto)

|  | previsto 0 | previsto 1 |
|---|---|---|
| **real 0** | TN (verdadeiro negativo) | FP (falso positivo) |
| **real 1** | FN (falso negativo) | TP (verdadeiro positivo) |

> Truque para não confundir: a **segunda palavra** é o que o modelo disse; a primeira diz se ele acertou. FP = disse positivo e errou.

### As métricas

| Métrica | Fórmula | Pergunta que responde | Quando importa |
|---|---|---|---|
| **Accuracy** | (TP + TN) / total | com que frequência acerta? | classes balanceadas |
| **Precision** | TP / (TP + FP) | quanto confiar num "positivo"? | falso positivo é caro (spam que apaga e-mail bom) |
| **Recall** | TP / (TP + FN) | quantos positivos reais encontra? | falso negativo é caro (doença não detectada) |
| **F1** | 2·P·R / (P + R) | equilíbrio entre P e R | precisa das duas altas |
| **ROC AUC** | área sob TPR × FPR | o modelo **ordena** bem positivos acima de negativos? | independe do limiar |

- **Accuracy engana** com classes desbalanceadas: sempre prever "normal" acerta 90% e não detecta nenhum spam.
- **Precision × recall** é um cabo de guerra controlado pelo **limiar**: baixar o limiar aumenta o recall e reduz a precision.
- **Por que média harmônica no F1?** Com P = 1,00 e R = 0,05, a média aritmética dá 0,525 (enganosa); o F1 dá 0,095.

### Multiclasse: one-vs-rest
Cada classe vira "positiva" uma vez e as outras viram "negativas". Na matriz k×k: **TP = diagonal**, **FP = resto da coluna**, **FN = resto da linha**, **support = soma da linha**.
- **macro avg:** média simples — todas as classes pesam igual.
- **weighted avg:** média ponderada pelo support.

### ROC e AUC
Para cada limiar: $TPR = \frac{TP}{TP+FN}$ (eixo Y) e $FPR = \frac{FP}{FP+TN}$ (eixo X). A curva vai de (0,0) (limiar infinito, nunca prevê positivo) a (1,1) (tudo positivo). AUC pela regra dos trapézios: $A = (x_2 - x_1)\cdot\frac{y_1 + y_2}{2}$.
AUC = 1: positivos sempre acima dos negativos · AUC = 0,5: ordena como uma moeda.

---

## Exercício guiado — 10 etapas ([`Exercicio Guiado - Aula 6.5.py`](Exercicio%20Guiado%20-%20Aula%206.5.py))

Antes das etapas marcadas com *preveja: ____*, **escreva o resultado esperado** e só depois execute.

| # | Etapa | Resultado com os dados do script |
|---|---|---|
| 01 | Carregar `y_real` e `y_pred` (10 mensagens, 1 = spam) | — |
| 02 | Acertos e accuracy | 7 acertos · accuracy **0,70** |
| 03 | Contar TP, TN, FP, FN com máscaras booleanas | TP 4 · TN 3 · FP 2 · FN 1 |
| 04 | Matriz de confusão | `[[3, 2], [1, 4]]` |
| 05 | Precision | **0,667** |
| 06 | Recall | **0,80** |
| 07 | F1 | **0,727** |
| 08 | Estender para 3 classes (Adelie, Chinstrap, Gentoo) | matriz 3×3 |
| 09 | `classification_report` próprio | precision/recall/F1/support + macro e weighted |
| 10 | ROC e AUC a partir das probabilidades | AUC **0,84** (vs. accuracy 0,70 com limiar 0,5) |

```bash
cd "Aula6/Aula 6.5"
python "Exercicio Guiado - Aula 6.5.py"
```

---

## Erros comuns

| Erro | Como corrigir |
|---|---|
| Confundir FP com FN | a 2ª palavra é o que o modelo disse |
| Confundir precision com recall | denominador = o que o modelo disse (precision) ou o que existe (recall)? |
| Accuracy como garantia | compare com a linha de base da classe majoritária |
| F1 como média aritmética | (0,8 + 0,5)/2 = 0,65 está errado; 2PR/(P+R) = 0,615 |
| Ignorar divisão por zero | TP + FP = 0 gera erro ou `nan`; trate e documente a convenção |
| Confundir macro e weighted | macro: classes pesam igual; weighted: pesa pelo support |
| Support = nº de previsões | support é a soma da **linha**: `M.sum(axis=1)` |
| ROC com rótulos binários | use `predict_proba(X)[:, 1]`, não `predict(X)` |
| Confundir AUC com accuracy | no exemplo: AUC 0,84 e accuracy 0,70 |
| Uma métrica só em base desbalanceada | mostre matriz, precision e recall |

## Desafio de casa — `meu_classification_report` do zero
Implementar `meu_classification_report(y_real, y_pred)` e, como extensão, `curva_roc()` e `auc()` com probabilidades.
**Critérios de aceite:** executa sem erros · matriz de confusão correta · métricas corretas · apresenta as classes existentes · support correto · médias macro e weighted corretas · relatório legível · coincide com o sklearn nas mesmas configurações · **nenhuma função pronta de métrica nos cálculos** · você explica cada métrica implementada.

## Leitura
scikit-learn — guia *Model evaluation* · `confusion_matrix` · `classification_report` · `precision_recall_fscore_support` · `roc_curve` · `roc_auc_score`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.5 - Metricas.pdf` | slides da aula |
| `Exercicio Guiado - Aula 6.5.py` | as 10 etapas, de contagens até ROC/AUC |

---

**← Anterior:** [Aula 6.4 — Regressão logística](../Aula%206.4/README.md) · **Próxima →** [Aula 6.6 — Validação cruzada](../Aula%206.6/README.md)
