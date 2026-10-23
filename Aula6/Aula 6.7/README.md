# Aula 6.7 — KNN do zero

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Python, NumPy e Matplotlib
> Material: [`Aula 6.7 KNN do Zero.pdf`](Aula%206.7%20KNN%20do%20Zero.pdf) (53 slides)

Na Aula 5.1 o KNN foi uma linha (`KNeighborsClassifier(n_neighbors=5)`). Agora a meta é **explicar como ele toma uma decisão**, não apenas executá-lo. O KNN é o oposto da regressão logística: em vez de resumir o treino em poucos coeficientes, ele **guarda todos os exemplos** e, na hora de prever, pergunta *"quem são os K exemplos mais parecidos com este?"* e deixa que eles votem. Toda a dificuldade está em calcular essas distâncias depressa — e é aí que o **broadcasting** do NumPy entra.

---

## Objetivos

| Entender | Calcular | Implementar |
|---|---|---|
| o significado de KNN e de K | distância euclidiana à mão | KNN com loops |
| como os vizinhos são escolhidos | distância de Manhattan à mão | KNN com broadcasting |
| votação majoritária e empates | matriz de distâncias com NumPy | a classe `KNNDoZero` |
| por que escalonar os dados | shapes de `X_train` e `X_test` | benchmark com `perf_counter` |
| broadcasting e vetorização | tempo e custo de memória | avaliação com acurácia |

---

## Teoria

### O que é KNN
**K** = quantos vizinhos consultamos · **Nearest** = os de menor distância · **Neighbors** = exemplos de treino.
É **aprendizado baseado em instâncias**: o `fit` só memoriza; o trabalho está no `predict`.

`receber a amostra → distância até cada exemplo → ordenar → separar os K mais próximos → rótulos → votação → classe`

### Distâncias

| | Euclidiana | Manhattan |
|---|---|---|
| Fórmula | $\sqrt{\sum (x_i - y_i)^2}$ | $\sum \lvert x_i - y_i\rvert$ |
| Geometria | linha reta (hipotenusa) | caminho em grade (quarteirões) |
| Exemplo A(1,2) → B(4,6) | **5** | **7** |

A métrica escolhida **pode mudar a classificação** do mesmo ponto.

### Votação e empates
```python
vizinhos = np.array([0, 1, 1, 0, 1])
np.bincount(vizinhos).argmax()   # → 1   (votos [2, 3])
```
K ímpar evita empates com 2 classes, mas não garante em multiclasse. Regra desta aula: **em empate, vence o menor rótulo** (determinística).

### Escolha de K
- **K pequeno** → decisão local, sensível a ruído, fronteiras irregulares → *overfitting*.
- **K grande** → decisão suave, pode perder padrões locais e favorecer a classe frequente → *underfitting*.
- Escolha por **validação cruzada** (Aula 6.6), testando K ∈ {1, 3, 5, 7, 9, 11}.

### Escalonamento
Cliente (25 anos, R$ 52.000): sem escala, a **renda domina** a distância e a idade não conta. Com `MinMaxScaler`, a decisão de K = 5 muda de "4 votos classe 1" para "5 votos classe 0". `fit` do escalonador **só no treino**; o teste recebe só `transform`.

### Broadcasting: a matriz de distâncias sem loops
```python
dif = X_test[:, None, :] - X_train[None, :, :]   # (n_test, 1, d) − (1, n_train, d) → (n_test, n_train, d)
D   = np.sqrt(np.sum(dif ** 2, axis=2))           # (n_test, n_train)
viz = np.argsort(D, axis=1, kind="stable")[:, :k] # índices dos k mais próximos
```
- `np.argsort` devolve **posições** (não distâncias); `np.argpartition` seleciona os K menores sem ordenar tudo.
- Custo: $O(n_{test}\cdot n_{train}\cdot d)$ em tempo, e o array intermediário `(n_test, n_train, d)` pode ocupar **gigabytes**.

### A classe `KNNDoZero`
```python
class KNNDoZero:
    def __init__(self, k=3, metrica="euclidiana"): ...   # valida k ≥ 1
    def fit(self, X, y): ...       # só guarda X_train e y_train
    def _distancias(self, X): ...  # broadcasting, euclidiana ou manhattan
    def predict(self, X): ...      # argsort → rótulos → bincount → argmax
    def score(self, X, y): ...     # acurácia
```

---

## Exercício guiado — 16 etapas
Notebook `Aula 6 7 Exercicio Guiado.ipynb`, dataset sintético com semente 42.

| Bloco | Etapas |
|---|---|
| Dados | 01 criar o dataset · 02 visualizar · 03 identificar X e y · 04 separar treino e teste |
| Distâncias | 05 euclidiana à mão · 06 Manhattan à mão · 07 função de distâncias |
| KNN | 08 seleção dos vizinhos · 09 votação · 10 KNN com loops · 11 KNN com broadcasting · 12 comparar previsões |
| Avaliação | 13 medir o tempo (`perf_counter`, 7 repetições, mediana) · 14 escalonar corretamente · 15 acurácia no teste · 16 fronteiras de decisão |

**Fronteiras de decisão** (80 exemplos, semente 42): K = 1 → treino 100%, fronteira irregular com "ilhas"; K intermediário → treino 88,8%, menos fragmentada; K = 9 → treino 82,5%, regiões suaves. Acertar 100% no treino com K = 1 é **memorização**, não qualidade.

---

## Erros comuns

| Erro | Por quê | Correto |
|---|---|---|
| Confundir KNN com K-Fold | K do KNN = vizinhos, não folds | `KNNDoZero(k=3)` / `KFold(n_splits=5)` |
| K maior que o treino | não existem K vizinhos | `assert 1 <= k <= len(X_train)` |
| Ordenar uma coluna em vez das distâncias | ordenar uma feature não mede proximidade | calcular `d` primeiro |
| `y_train[np.sort(d)[:k]]` | `sort` devolve distâncias, não posições | `np.argsort(d)[:k]` |
| `np.argsort(d)[-k:]` | pega os mais **distantes** | `[:k]` |
| `axis` incorreto | soma sobre os treinos, não sobre as features | `np.sum(dif ** 2, axis=2)` |
| `X_test - X_train` | `(3, 2) − (5, 2)` não combinam | `X_test[:, None, :] - X_train[None, :, :]` |
| Escalonar o teste à parte ou antes de dividir | parâmetros diferentes / vazamento | `fit` no treino, `transform` no teste |
| `np.argmax(vizinhos)` | devolve uma **posição** | `np.bincount(vizinhos).argmax()` |
| Desempatar aleatoriamente | resultado muda a cada execução | regra fixa (`min` das empatadas) |
| Medir tempos com dados diferentes | mede o tamanho, não a técnica | mesmos dados nas duas versões |
| Achar que broadcasting é "sem custo" | o intermediário pode ter gigabytes | processar em blocos |
| Avaliar só no treino | com K = 1 acerta 100% por memorização | avaliar no teste |

## Desafio de casa — KNN do zero, loops × broadcasting
Implementação com: diferentes K · euclidiana e Manhattan · seleção de K vizinhos · votação majoritária · previsões · acurácia · múltiplas features · regra explícita e determinística para empates. **Versão 1** com loops e **versão 2** com broadcasting (notebook `Aula 6 7 Desafio Final.ipynb`, com TODOs).

**Entregas:** código das duas versões · prova de previsões equivalentes · tempos com `perf_counter` · gráfico comparativo · interpretação · com e sem escalonamento · fronteiras com diferentes K · conclusão sobre vantagens e limitações.
**Critérios de aceite:** distâncias corretas · vizinhos = menores distâncias · votação correta · empates consistentes · escalonamento sem informação do teste · comparação justa · tempos realmente medidos · gráfico e conclusão.

## Leitura
NumPy: *Broadcasting*, `argsort`, `argpartition`, indexação, *quickstart* · scikit-learn: `KNeighborsClassifier`, `StandardScaler`, `MinMaxScaler`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.7 KNN do Zero.pdf` | slides da aula (o código do exercício guiado está nos slides 29–47) |

> Os notebooks `Aula 6 7 Exercicio Guiado.ipynb` e `Aula 6 7 Desafio Final.ipynb` citados nos slides ainda não estão nesta pasta.

---

**← Anterior:** [Aula 6.6 — Validação cruzada](../Aula%206.6/README.md) · **Próxima →** [Aula 6.8 — Naive Bayes do zero](../Aula%206.8/README.md)
