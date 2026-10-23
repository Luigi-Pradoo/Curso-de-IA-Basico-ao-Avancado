# Aula 6.2 — Regressão linear pela equação normal

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Dataset: **California Housing**
> Material: [`6.2 - Regressão linear pela equação normal.pdf`](6.2%20-%20Regress%C3%A3o%20linear%20pela%20equa%C3%A7%C3%A3o%20normal.pdf) (43 slides) + [`ExercicioGuiado.py`](ExercicioGuiado.py)

Na 6.1 chegamos ao gradiente do MSE. Hoje fazemos a pergunta natural: **onde ele vale zero?** Igualando $\nabla J = 0$ e isolando os coeficientes sai uma fórmula fechada — a **equação normal** — que encontra a melhor reta (ou hiperplano) numa única conta:

$$\hat\beta = (X^TX)^{-1}X^Ty$$

A aula vai do gráfico de pontos até a sua própria classe `MinimosQuadrados`, passando pelos três jeitos de resolver isso no NumPy (`inv`, `solve`, `pinv`) e por quando cada um falha.

---

## Objetivos

1. Entender a regressão linear como uma **regra de previsão** $\hat y = \beta_0 + \beta_1 x$.
2. Definir resíduo e explicar por que somamos os erros **ao quadrado**.
3. Escrever o problema em forma matricial (`y`, `X` com coluna de 1s, `β`).
4. Deduzir a equação normal em cinco passos.
5. Resolver com `np.linalg.inv`, `solve` e `pinv` — e saber quando cada um quebra.
6. Calcular o R² e conferir o resultado com o `LinearRegression`.

---

## Teoria (blocos A–D)

### A — Entendendo o problema
Quatro casas vendidas (área → preço). Queremos uma regra que estime o preço de qualquer outra.
- **Resíduo:** $e_i = y_i - \hat y_i$ (positivo quando o ponto está acima da reta).
- **Por que ao quadrado?** sem cancelamento entre erros + e −, erros grandes pesam mais, e a curva de custo é lisa (derivável).
- Variando $\beta_1$ o custo é uma **parábola**; variando $\beta_0$ e $\beta_1$ juntos, uma **tigela** com um único fundo.

### B — Da reta para as matrizes
Cada casa é uma equação $y_i = \beta_0\cdot 1 + \beta_1 x_i + e_i$. Empilhando:
- `y` (4×1), **`X` (4×2) = [coluna de 1s | área]**, `β` (2×1).
- `X @ β` calcula **todas as previsões de uma vez**.
- A coluna de 1s é o que faz o intercepto $\beta_0$ entrar na conta.

### C — Dedução em cinco passos
1. **Custo:** $J(\beta) = (y - X\beta)^T(y - X\beta)$
2. **Expandir:** $y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$
3. **Derivar:** $\nabla J = -2X^Ty + 2X^TX\beta$
4. **Igualar a zero:** $X^TX\beta = X^Ty$ ← *equações normais*
5. **Isolar:** $\beta = (X^TX)^{-1}X^Ty$

**Leitura geométrica:** a previsão $X\beta$ é a **projeção (sombra)** de `y` no plano gerado pelas colunas de `X`; o resíduo fica em ângulo reto com elas ($X^Te = 0$).

**Exemplo das quatro casas:** $\beta = [0{,}5;\ 1{,}6]$, previsões `[2,1  3,7  5,3  6,9]`, **R² = 0,985**.

**Três perguntas diferentes:** existe solução? (sempre) · é única? (só se as colunas de X forem independentes) · é estável? (depende do condicionamento).

### D — Inversa, solve e pseudoinversa

| Método | Como funciona | Com $X^TX$ singular |
|---|---|---|
| `np.linalg.inv(X.T @ X) @ X.T @ y` | monta a inversa inteira — a fórmula literal | `LinAlgError` ou números enormes sem aviso |
| `np.linalg.solve(X.T @ X, X.T @ y)` | resolve o sistema sem formar a inversa | `LinAlgError` |
| `np.linalg.pinv(X) @ y` | pseudoinversa via SVD ($X = U\Sigma V^T$) | **sempre responde**, com a solução de **menor norma** |

Com uma coluna duplicada (`X_dup = [1, x, x]`) há infinitas soluções; a `pinv` divide o 1,6 igualmente: (0,8; 0,8).

---

## Exercício guiado — California Housing ([`ExercicioGuiado.py`](ExercicioGuiado.py))

| # | Passo | Código |
|---|---|---|
| 1 | Carregar e conferir formatos | `X (20640, 8)`, `y (20640,)` |
| 2 | Adicionar a coluna de 1s | `Xb = np.column_stack((np.ones(n), X))` → 9 colunas |
| 3 | Calcular β pelos três métodos | `inv`, `solve`, `pinv` + `np.allclose` |
| 4 | Ler os coeficientes | `MedInc` ≈ +0,44 (+US$ 10 mil de renda → +US$ 44 mil no preço) |
| 5 | R² e conferência com `LinearRegression` | R² ≈ 0,61 e coeficientes iguais aos do sklearn |

$$R^2 = 1 - \frac{SQ_{res}}{SQ_{tot}} = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$$
(R² = 1: previsão perfeita · R² = 0: igual a chutar sempre a média · R² < 0: pior que a média)

```bash
cd "Aula6/Aula 6.2"
python ExercicioGuiado.py
```

---

## Erros comuns
1. **Esquecer a coluna de 1s** — roda sem erro, mas a reta passa à força pela origem.
2. **Formatos incompatíveis** — `X @ X` com `(4,2)@(4,2)`; use `X.T @ X`.
3. **`*` no lugar de `@`** — devolve `(4, 2)` em vez de 4 previsões.
4. **Matriz singular** — colunas repetidas/proporcionais ou todas as dummies junto com a coluna de 1s.
5. **Inversa sem necessidade** — prefira `solve` ou `pinv`.
6. **Escalas muito diferentes** (`Population` × `AveBedrms`) — o número de condição cresce; padronize com média e desvio do treino.
7. **Intercepto em dobro** — passar `Xb` (com 1s) ao `LinearRegression`; passe `X` sem 1s ou use `fit_intercept=False`.

## Desafio de casa — classe `MinimosQuadrados`

```python
MinimosQuadrados(metodo="pinv")   # "inv", "solve" ou "pinv"; outro valor → ValueError
.fit(X, y)       # adiciona 1s, calcula β, guarda em self.coef_, devolve self
.predict(X)      # mesmo tratamento do fit, devolve ŷ
.score(X, y)     # devolve o R²
```
Só NumPy, nenhum `for` sobre linhas ou colunas.

| Etapa | Ponto de controle |
|---|---|
| `__init__` | `MinimosQuadrados("abc")` gera `ValueError` |
| `_adicionar_intercepto` | `X (4, 1)` vira `(4, 2)`, primeira coluna só 1s |
| `fit` com `"solve"` | quatro casas → `coef_ = [0.5, 1.6]` |
| `fit` com `"inv"` e `"pinv"` | os três métodos dão o mesmo `coef_` |
| `predict` | `[2.1, 3.7, 5.3, 6.9]` |
| `score` | `0.985` |
| California Housing | coeficientes, previsões e R² batem com o `LinearRegression`; com `X_dup`, mostrar o que `inv`/`solve` fazem e que `pinv` ainda funciona; imprimir o número de condição nos dois casos |

## Leitura
`numpy.linalg.lstsq` · `numpy.linalg.pinv` · `numpy.linalg.cond` · `numpy.linalg.solve` · `LinearRegression` (scikit-learn).

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `6.2 - Regressão linear pela equação normal.pdf` | slides da aula |
| `ExercicioGuiado.py` | equação normal no California Housing, conferida com o sklearn |

---

**← Anterior:** [Aula 6.1 — Álgebra linear e cálculo](../Aula%206.1/README.md) · **Próxima →** [Aula 6.3 — Gradiente descendente](../Aula%206.3/README.md)
