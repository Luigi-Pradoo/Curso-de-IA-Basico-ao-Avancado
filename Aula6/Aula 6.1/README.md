# Aula 6.1 — Álgebra linear e cálculo mínimos para ML

> **Módulo 06 · ML na mão, com NumPy** · Aula 1 de 13 · 2 h
> Material: [`6.1 - Álgebra linear e cálculo mínimos para ML.pdf`](6.1%20-%20%C3%81lgebra%20linear%20e%20c%C3%A1lculo%20m%C3%ADnimos%20para%20ML.pdf) (47 slides) + [`Desafio.py`](Desafio.py)

Só o necessário de matemática para ler o que acontece dentro de `model.fit(X, y)`: **vetores, matrizes, produto interno, norma, derivada parcial e gradiente** — tudo em NumPy vetorizado, sem loops. A aula sobe em quatro níveis e termina no resultado que sustenta as próximas três aulas: o **gradiente do erro quadrático médio** escrito numa linha.

```text
Nível 4  Gradiente do MSE      ∇L = (2/n) Xᵀ(Xw − y)      ← tudo junto numa linha
Nível 3  Cálculo               derivada, regra da cadeia, gradiente
Nível 2  Matrizes              X @ w, transposta, inversa
Nível 1  Vetores               produto interno, norma, distância
```

---

## Objetivos

1. Representar amostras, datasets e pesos como arrays e **prever o shape** de qualquer operação entre eles.
2. Calcular produto interno, normas L1 e L2 e distância entre vetores — à mão e em NumPy.
3. Ler `X @ w` de duas formas (por linhas e por colunas).
4. Calcular derivadas numéricas e aplicar a regra da cadeia.
5. Derivar o gradiente do MSE na forma matricial e conferi-lo numericamente.
6. Reescrever loops como operações matriciais e medir o ganho de tempo.

---

## Teoria

### Nível 1 — vetores
| Objeto | Exemplo no Iris | `ndim` | `shape` |
|---|---|---|---|
| Escalar | 3.5 | 0 | `()` |
| Vetor | uma flor | 1 | `(4,)` |
| Matriz | o Iris inteiro | 2 | `(150, 4)` |
| Tensor | 32 imagens 28×28 | 3 | `(32, 28, 28)` |

- **Eixos:** o `axis` que você passa é o que **desaparece** do shape (`M.sum(axis=0)` colapsa as linhas).
- **Fatias são *views*:** alterar a fatia altera o original — use `.copy()`.
- **Broadcasting:** `X (150, 4) − mu (4,) → (150, 4)`.
- **Produto interno:** $x \cdot w = \sum_i x_i w_i$ → `x @ w`. Geometricamente $x \cdot w = \|x\|\|w\|\cos\theta$ (>0 agudo, =0 ortogonais, <0 obtuso).
- **Similaridade de cosseno** e **projeção** $\text{proj}_w(x) = \frac{x\cdot w}{w\cdot w}\,w$.
- **Normas:** L2 `[3,4] → 5`, L1 `[3,4] → 7`, L∞ `[3,4] → 4`; distância $d(a,b) = \|a-b\|$.

### Nível 2 — matrizes
- Convenção do módulo: **n amostras nas linhas, d features nas colunas**.
- **Regra dos shapes:** as dimensões de dentro precisam ser iguais; as de fora formam o resultado. `X @ w`: `(n, d) @ (d,) → (n,)` — as *n* previsões de uma vez.
- **Duas leituras de `X @ w`:** cada saída é o produto interno de uma linha com `w` **ou** uma combinação das colunas de `X` com pesos `w`.
- Matriz como **transformação do espaço**; transposta $(AB)^T = B^TA^T$; quando **não existe inversa** (determinante 0, colunas dependentes).

### Nível 3 — cálculo
- **Derivada numérica:** diferença progressiva (erro ~h) × **central** (erro ~h²) — `(f(x+h) − f(x−h)) / 2h`.
- Regras usadas no módulo: constante, soma, potência, produto, **cadeia**; destaque para a sigmoide $\sigma'(z) = \sigma(z)(1-\sigma(z))$ (Aula 6.4).
- **Derivada parcial e gradiente:** $f(w_1,w_2) = w_1^2 + 3w_2^2 \Rightarrow \nabla f = [2w_1,\ 6w_2]$.
- O gradiente aponta para onde a função **mais sobe**; para descer, ande contra ele: $w \leftarrow w - \eta\nabla f$.

### Nível 4 — gradiente do MSE
$$e = Xw - y,\qquad L(w) = \tfrac{1}{n}\textstyle\sum e_i^2,\qquad \nabla L = \tfrac{2}{n}\,X^T(Xw - y)$$

```python
grad = 2 / n * X.T @ (X @ w - y)     # (d, n) @ (n,) → (d,)
```

Igualar esse gradiente a zero e resolver em `w` é exatamente a **Aula 6.2**.

---

## Exercício guiado — reescrever loops como matrizes

Dados sintéticos com `n = 10.000`, `d = 5`, semente 42. Para cada tarefa: versão com laço, versão vetorizada, `np.allclose` e `%timeit`.

1. Produto interno de uma amostra
2. Todas as previsões (`X @ w`)
3. Distâncias entre pares de pontos (`(200, 50)`)
4. MSE e gradiente
5. Conferir o gradiente com a diferença central e cronometrar

## Erros comuns
| Erro | O que acontece | Correção |
|---|---|---|
| `*` no lugar de `@` | `X * w` multiplica elemento a elemento e devolve `(n, d)` **sem erro nenhum** | `X @ w` |
| `(n,)` contra `(n, 1)` | `y.reshape(-1, 1) - y_hat` vira uma matriz `(n, n)` em silêncio | cheque o shape antes da média |
| `axis` trocado | `X.mean(axis=0)` é por feature; `axis=1` é por amostra | pense em qual eixo deve sumir |
| Esquecer `keepdims` | `X.sum(axis=1)` tem shape `(n,)` | `keepdims=True` → `(n, 1)` para subtrair por linha |
| *View* alterando o original | alterar a fatia altera o array de origem | `.copy()` |
| Inversa para resolver sistema | `np.linalg.inv` é instável | `solve` ou `lstsq` |
| Broadcasting que estoura a memória | `(10000, 1, 50) − (1, 10000, 50)` = 5 bilhões de floats (40 GB) | truque $\|a\|^2 + \|b\|^2 - 2a\cdot b$ |
| `h` pequeno demais | com `h = 1e-15` o arredondamento domina | diferença central com `h` entre `1e-4` e `1e-6` |

---

## Desafio de casa — módulo `algebra.py` ([`Desafio.py`](Desafio.py))

Implementar **só com NumPy**, sem laços sobre amostras:

| Função | O que faz |
|---|---|
| `norma(x, ord=2)` | normas L1 e L2 (`ValueError` para outros `ord` ou se `x` não for vetor) |
| `distancias(A, B)` | matriz `(m, k)` de distâncias euclidianas, via $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a\cdot b$ |
| `mse(X, y, w)` | erro quadrático médio |
| `grad_mse(X, y, w)` | gradiente analítico |
| `grad_numerico(f, w, h)` | gradiente por diferença central |

**Critérios de aceite:** nenhum `for`/`while` fora de `grad_numerico` · `norma` bate com `np.linalg.norm` e `distancias` com `scipy.spatial.distance.cdist` (`np.allclose`) · `grad_mse` e `grad_numerico` concordam em cinco sementes · tudo roda em menos de 1 s com `X` de shape `(10000, 50)`.

O script já roda 7 testes: normas contra `np.linalg.norm`, distâncias contra `scipy.cdist`, MSE, gradiente analítico × numérico, **cinco sementes aleatórias** e desempenho (`X` de 10.000 × 50 em menos de 1 s).

```bash
cd "Aula6/Aula 6.1"
python Desafio.py
```

## Aprofundamento (slides A1–A6)
Hessiana e convexidade ($H = \frac{2}{n}X^TX$ é semidefinida positiva) · autovalores e autovetores · número de condição $\kappa = \lambda_{max}/\lambda_{min}$ (descida em zigue-zague, inversa instável) · `np.einsum` · distâncias sem estourar a memória · checagem de gradiente por erro relativo.

## Leitura
NumPy: *the absolute basics for beginners* · *Broadcasting* · `numpy.matmul` · `numpy.linalg` · `numpy.einsum`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `6.1 - Álgebra linear e cálculo mínimos para ML.pdf` | slides da aula |
| `Desafio.py` | solução do desafio `algebra.py` com os testes |

---

**← Anterior:** [Aula 5.5 — Seleção de features](../../Aula5/Aula%205.5/README.md) · **Próxima →** [Aula 6.2 — Equação normal](../Aula%206.2/README.md)
