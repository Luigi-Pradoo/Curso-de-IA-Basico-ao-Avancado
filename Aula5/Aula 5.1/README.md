# Aula 5.1 — O pipeline completo de ML

> **Módulo 05 · Fundamentos práticos do ML** · 2 h · Dataset: **Iris** · scikit-learn
> Material: [`Aula 5.1 - ML.pdf`](Aula%205.1%20-%20ML.pdf) (39 slides)

Primeiro modelo do curso. Pegamos o Iris que exploramos na Aula 4 e percorremos o ciclo inteiro de um projeto **supervisionado**: separar entradas (`X`) e resposta (`y`), dividir treino e teste, treinar um KNN, prever, medir e — o passo que quase todo mundo esquece — **comparar com um baseline**. Um número só faz sentido quando sabemos com o que compará-lo.

---

## Objetivos

1. Explicar o que é aprendizado supervisionado.
2. Identificar features, alvo, classes e amostras.
3. Reconhecer a estrutura `X` (matriz) e `y` (vetor).
4. Executar um pipeline básico de ponta a ponta.
5. Separar dados de treino e de teste.
6. Treinar um classificador no scikit-learn.
7. Gerar previsões para dados novos.
8. Interpretar accuracy e matriz de confusão.
9. Comparar o modelo com um baseline.

---

## Teoria

**Machine learning:** o modelo encontra padrões nos dados e usa esses padrões para prever.
`DADOS → APRENDIZADO → MODELO → PREVISÃO`

**Supervisionado:** cada exemplo vem com a resposta certa. Se `y` é categoria → **classificação** (hoje); se é número contínuo → **regressão**.

**Iris:** `X.shape == (150, 4)` (comprimento/largura de sépala e pétala, em cm) e `y.shape == (150,)` (setosa, versicolor, virginica — 50 de cada).

### As oito etapas de qualquer projeto supervisionado
`DADO → EXPLORAR → X/y → SPLIT → FIT → PREDICT → AVALIAR → MÉTRICA`

O **split** é a caixa que separa um número honesto de um número bonito.

### Dois argumentos que não se esquecem

```python
train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
```

| Argumento | O que faz | Sem ele… |
|---|---|---|
| `random_state=42` | fixa o sorteio → mesmo split, mesmo número no fim | cada execução conta uma história diferente |
| `stratify=y` | mantém a proporção das classes (15 de cada espécie no teste) | a proporção varia e a comparação fica injusta |

### Como o KNN decide
`ponto novo → distâncias até todo o treino → os K=5 mais perto → voto → classe`

O `fit` do KNN **só guarda a tabela**; todo o trabalho acontece no `predict`. "Perto" é a distância euclidiana — Pitágoras com 4 medidas:

$$d(x, y) = \sqrt{\textstyle\sum_i (x_i - y_i)^2}$$

Por isso **escala importa** (assunto da Aula 5.4).

### Avaliar
- **Accuracy** = previsões corretas / total → **43 de 45 = 95,6%**.
- **Matriz de confusão**: diagonal = acerto, fora da diagonal = erro. Os dois erros ficam entre *versicolor* e *virginica*, as espécies que se sobrepõem.
- **Baseline** (`DummyClassifier(strategy="most_frequent")`): **33,3%**. É contra isso que os 95,6% devem ser lidos.

---

## Exercício guiado (7 etapas)

| # | Etapa | Resultado esperado |
|---|---|---|
| 1 | Carregar e separar `X` e `y` | `(150, 4) (150,)` |
| 2 | Explorar (`groupby("especie").mean()`) | pétala separa melhor as espécies |
| 3 | `train_test_split(..., stratify=y, random_state=42)` | `(105, 4) (45, 4)` |
| 4 | `KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)` | — |
| 5 | `predict(X_te)` | previsões para as 45 flores |
| 6 | `accuracy_score` e `confusion_matrix` | `0.9556` |
| 7 | `DummyClassifier` | `0.3333` |

O pipeline inteiro numa tela:

```python
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

modelo = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
y_pred = modelo.predict(X_te)

print(accuracy_score(y_te, y_pred))
print(confusion_matrix(y_te, y_pred))
print(DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).score(X_te, y_te))
```

---

## Erros comuns

1. **Prever antes do `fit`** → `NotFittedError`.
2. **Medir no treino** → `score(X_tr, y_tr)` mede memória. Reporte sempre o teste.
3. **Esquecer `random_state`** → outro número a cada execução.
4. **Esquecer `stratify`** → a proporção das classes muda.
5. **Escolher K olhando o teste** → transforma teste em treino.
6. **Transformar antes de separar** → informação do teste vaza para o treino (Aula 5.2).

## Desafio de casa

Refaça as sete etapas com outro dataset de classificação do scikit-learn — `load_wine` ou `load_breast_cancer`. Mesmo modelo, mesmo K, mesma ordem de células.
**Entrega:** notebook com accuracy de teste, matriz de confusão, comparação com o `DummyClassifier` e três linhas explicando o resultado com suas palavras.

## Leitura
- `load_iris` · `train_test_split` · `KNeighborsClassifier` · `accuracy_score` / `confusion_matrix` — documentação do scikit-learn.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 5.1 - ML.pdf` | slides da aula (o código do exercício está nos slides 27–34) |

---

**← Anterior:** [Aula 4 — EDA](../../Aula4/README.md) · **Próxima →** [Aula 5.2 — Treino, validação e teste](../Aula%205.2/README.md)
