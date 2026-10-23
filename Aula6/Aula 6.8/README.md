# Aula 6.8 — Naive Bayes do zero

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Projeto: **filtro de spam** em português
> Material: [`Aula 6.8 - Naive Bayes do Zero.pdf`](Aula%206.8%20-%20Naive%20Bayes%20do%20Zero.pdf) (52 slides)

*"Parabéns! Você ganhou uma promoção!"* — isso é spam? A ideia do Naive Bayes é simples: comparar **com que frequência cada palavra aparece** em mensagens de spam e em mensagens normais já rotuladas, e transformar essas contagens numa decisão com o **Teorema de Bayes**. Diferente da regressão logística e do KNN, aqui não há gradiente nem distância: o "treino" é **contar**. A aula vai da probabilidade básica até um classificador completo em NumPy, passando pelos dois truques que fazem o método funcionar na prática — **suavização de Laplace** e **logaritmos**.

```text
Mensagem → contagens de palavras → pontuação por classe (spam / não spam) → argmax → classe
```

---

## Objetivos

| Probabilidade | Texto | O modelo | Implementação |
|---|---|---|---|
| probabilidade e probabilidade condicional | textos como contagens de palavras | o significado de "Naive" | classificador em NumPy |
| Teorema de Bayes | suavização de Laplace | independência condicional à classe | filtro de spam do zero |
| priori, verossimilhança e posteriori | logaritmos contra o *underflow* | Gaussian e Multinomial | acurácia e matriz de confusão |

---

## Teoria

### Probabilidade condicional
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
O "|" se lê **"dado que"**: a condição define o universo da contagem. Com 100 mensagens (40 spam; 36 com "grátis", das quais 30 são spam):
- $P(\text{Grátis} \mid \text{Spam}) = 30/40 = 0{,}75$ — contamos só entre os spams.
- $P(\text{Spam} \mid \text{Grátis}) = 30/36 = 0{,}8333$ — contamos só entre as que têm "grátis".

**$P(A \mid B) \neq P(B \mid A)$.**

### Teorema de Bayes
$$\underbrace{P(C \mid x)}_{\text{posteriori}} = \frac{\overbrace{P(x \mid C)}^{\text{verossimilhança}}\ \cdot\ \overbrace{P(C)}^{\text{priori}}}{\underbrace{P(x)}_{\text{evidência}}}$$
- **Priori** — crença antes de ler a mensagem (proporção de spam).
- **Verossimilhança** — quão comum é a evidência dentro da classe.
- **Posteriori** — crença depois de ler.

### Por que "Naive" (ingênuo)
Supõe **independência condicional à classe**: dentro de cada classe, as palavras são independentes.
$$P(\text{grátis}, \text{promoção} \mid \text{Spam}) = P(\text{grátis} \mid \text{Spam}) \times P(\text{promoção} \mid \text{Spam})$$
É falso (as palavras andam juntas), mas funciona bem: sem a hipótese seriam $2^n$ parâmetros por classe (n = 20 → mais de 1 milhão); com ela, só **n**. Funciona com poucos dados e é rápido.

**Classificar:** $\hat y = \arg\max_C\ P(C)\prod_i P(x_i \mid C)$ — o denominador é igual para todas as classes e pode ser ignorado para decidir.

### Duas variantes

| | Gaussian NB | Multinomial NB |
|---|---|---|
| Dado | contínuo (altura, peso, temperatura) | contagens não negativas (palavras) |
| Modelo por classe | média e variância de cada feature → densidade normal | frequência de cada palavra |
| Uso típico | sensores, medidas | texto, *bag of words* |

### Do texto à matriz: *Bag of Words*
Tokenizar (minúsculas, só letras) → vocabulário **só do treino** → cada mensagem vira um vetor de contagens `X (n, |V|)`. A ordem das palavras é descartada.

### Suavização de Laplace
Sem suavização, "reunião grátis" zera a pontuação do spam porque "reunião" nunca apareceu em spam (um único fator 0 anula o produto).
$$P(w \mid C) = \frac{N_{w,C} + \alpha}{N_C + \alpha\,|V|}$$
Com α = 1: $P(\text{grátis} \mid \text{Spam}) = (2+1)/(5+6) = 3/11$. Nenhuma palavra fica com probabilidade zero e as probabilidades ainda somam 1.

### Logaritmos contra o *underflow*
Multiplicar centenas de probabilidades pequenas dá um número menor que o `float64` consegue representar (≈ 5×10⁻³²⁴) → vira 0. Como $\log(a \cdot b) = \log a + \log b$ e o log é crescente, **somamos logs** e o argmax não muda:
$$\text{log-pontuação}(C) = \log P(C) + \sum_i n_i \log P(w_i \mid C)$$

```python
log_scores = np.array([-3.2917, -5.2983])   # "grátis promoção": spam, normal
m = log_scores.max()                         # subtrair o máximo: normalização estável
probs = np.exp(log_scores - m)
probs = probs / probs.sum()                  # [0.8815, 0.1185] → spam
```

---

## Exercício guiado — 12 etapas da matemática ao código

| Vetorização (fora do modelo) | Treino (`fit`) | Predição e avaliação |
|---|---|---|
| 1 preparar mensagens · 2 tokenizar · 3 vocabulário **só do treino** · 4 vetorizar contagens `X (n, \|V\|)` | 5 rótulos (`np.unique`) · 6 prioris $P(C)$ · 7 contar palavras por classe · 8 Laplace · 9 log-probabilidades | 10 `X @ log_prob.T + log_priori` · 11 argmax · 12 acurácia e matriz de confusão |

Tudo organizado na classe `NaiveBayesDoZero(alpha=1.0)` com `fit`, `predict` e `predict_proba`.

**Dataset:** 32 mensagens (16 spam, 16 normais); a cada 4 mensagens uma vai para o teste → treino 12/12, teste 4/4.

**Resultado no teste:** acurácia **7/8 = 0,875** · precisão spam 4/5 = 0,8 · recall spam 4/4 = 1,0. O único erro: *"Parabéns você ganhou o prêmio de melhor projeto"* (normal) classificada como spam com P = 0,757 — as palavras de spam pesaram mais que o contexto. No treino o acerto é 24/24: avaliar só no treino seria otimista.

**Mensagem nova** — *"Parabéns! Você ganhou uma promoção!"*: log-pontuação spam −17,05 > normal −20,88 → **spam**, P(spam | msg) ≈ 0,979. A palavra "uma" está fora do vocabulário e é ignorada.

**Palavras que mais pesam:** "ganhe" (+1,797 → ~6× mais provável em spam), "clique", "grátis", "promoção" puxam para spam; "amanhã", "reunião", "aula" puxam para normal.

---

## Erros comuns

| # | Erro | Consequência |
|---|---|---|
| 1 | Confundir $P(A\mid B)$ com $P(B\mid A)$ | ler 0,75 = P(Grátis\|Spam) como "75% de spam" |
| 2 | Confundir priori e posteriori | decidir só pela proporção das classes |
| 3 | Tratar a independência condicional como absoluta | explicação errada do modelo e de seus limites |
| 4 | Trocar Gaussian por Multinomial | cada variante supõe uma distribuição |
| 5 | Frequências incorretas (dividir por nº de mensagens ou por \|V\|) | probabilidades erradas |
| 6 | Contar documentos em vez de tokens | 0,5 em vez de 0,4 para "grátis" |
| 7 | Esquecer a suavização de Laplace | produto 0, `log(0) = -inf`, empates |
| 8 | Denominador de Laplace errado (somar α só no numerador) | probabilidades somam mais que 1 |
| 9 | Multiplicar muitas probabilidades | pontuações 0.0 por *underflow* |
| 10 | Ler log-probabilidade como probabilidade | "−3,29 de chance" não existe |
| 11 | Vocabulário construído com o teste | vazamento → avaliação otimista |
| 12 | Características negativas no Multinomial | modelo matematicamente inválido (exige contagens ≥ 0) |
| 13 | Ordem do vocabulário diferente entre treino e teste | probabilidades aplicadas à palavra errada |
| 14 | Palavra desconhecida sem regra | `KeyError` ou previsões inconsistentes |
| 15 | Avaliar só no treino | 24/24 esconde o erro do teste |

## Desafio de casa — filtro de spam com Naive Bayes do zero
Novo dataset: **48 mensagens em português (24 spam, 24 normais)**, diferentes das do exercício. Você recebe o notebook com enunciado, esqueleto da classe, TODOs e testes automáticos.

| Dados | Modelo | Avaliação |
|---|---|---|
| 1 criar/carregar o dataset · 2 separar treino e teste · 3 vocabulário só com o treino · 4 vetores de contagem | 5 prioris · 6 frequências · 7 Laplace · 8 log-pontuações · 9 `fit()` · 10 `predict()` · 11 `predict_proba()` | 12 acurácia e matriz de confusão · 13 testar mensagens novas · 14 explicar duas previsões |

**Extras opcionais:** remover *stopwords*, variar α e comparar com o `MultinomialNB`.
**Critérios de aceite:** executa sem erros · probabilidades e suavização corretas · sem vazamento entre treino e teste · previsões = maiores pontuações · `predict_proba` soma 1 por linha · funciona com mensagens novas · métricas coerentes com as previsões · você explica cada etapa.
**Teste rápido do núcleo:** no mini dataset, *"grátis promoção"* deve dar **P(spam) = 0,8815**.
**Entrega:** notebook preenchido e executado, com um parágrafo sobre os erros do modelo.

## Leitura
NumPy: `numpy.log` · scikit-learn: *Naive Bayes*, `GaussianNB`, `MultinomialNB`, extração de características de texto (`CountVectorizer`).

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.8 - Naive Bayes do Zero.pdf` | slides da aula |

> ⚠️ Este PDF foi exportado em página A4 **retrato**, com cada slide **deitado (girado 90°)**. O conteúdo está íntegro; para ler, gire a visualização no leitor de PDF ou reexporte em formato paisagem 16:9.

---

**← Anterior:** [Aula 6.7 — KNN do zero](../Aula%206.7/README.md) · **Voltar ao módulo:** [Aula 6](../README.md)
