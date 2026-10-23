# Aula 6.4 — Regressão logística do zero

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Dataset sintético (2 classes, 100 amostras)
> Material: [`Aula 6.4 Regressao Logistica.pdf`](Aula%206.4%20Regressao%20Logistica.pdf) (48 slides) + [`ExercicioGuiado.py`](ExercicioGuiado.py)

Até aqui previmos **números** (preço de casa). Agora a resposta é **0 ou 1** — spam ou não, doente ou saudável. A regressão linear não serve: ela devolve valores acima de 1 e abaixo de 0, que não são probabilidades. A regressão logística mantém a combinação linear $z = X\beta$, mas passa o resultado pela **sigmoide**, que espreme qualquer número no intervalo (0, 1). O treino é o mesmo gradiente descendente da 6.3 — só muda o custo, que agora é a **entropia cruzada**.

```text
x  ──►  z = Xβ (escore)  ──►  p = σ(z) (probabilidade)  ──►  ŷ = 1 se p ≥ 0,5 (classe)
```

---

## Objetivos

1. Explicar por que a regressão linear não serve para classificar em 0 ou 1.
2. Transformar as entradas em probabilidade usando uma combinação linear e a sigmoide.
3. Calcular a entropia cruzada à mão e interpretar cada termo.
4. Deduzir o gradiente do custo e usá-lo para atualizar os parâmetros.
5. Implementar, treinar e avaliar o classificador em NumPy e desenhar sua fronteira de decisão.

---

## Teoria

### Sigmoide
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| z | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| σ(z) | 0,018 | 0,119 | **0,5** | 0,881 | 0,982 |

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Três números diferentes:** `z` (escore, qualquer real) · `p` (probabilidade estimada) · `ŷ` (classe prevista após o limiar) — e `y`, o rótulo verdadeiro.

**Limiar:** `ŷ = 1` se `p ≥ t`. O padrão é `t = 0,5`, mas ele é escolhido conforme o problema (Aula 6.5).

### Entropia cruzada binária
$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]$$

Se `y = 1` a perda é $-\log p$; se `y = 0`, $-\log(1-p)$. Mede **quão pouca probabilidade o modelo deu à classe correta**.

**Por que não o erro quadrático?** Com `y = 1` e `p = 0,01` (confiante e errado), o erro quadrático dá 0,98, enquanto a entropia cruzada dá **4,6** — ela pune muito mais a confiança errada, e o custo fica convexo.

**De onde ela vem:** maximizar a **verossimilhança** $\prod p_i^{y_i}(1-p_i)^{1-y_i}$; o log transforma produto em soma e evita *underflow*.

### Gradiente
Com $\sigma'(z) = \sigma(z)(1-\sigma(z))$ e a regra da cadeia, o fator $p(1-p)$ se cancela:
$$\nabla J(\beta) = \frac{1}{m}X^T(p - y)$$

```python
gradiente = X.T @ (p - y) / m
beta = beta - alpha * gradiente
```
É **a mesma forma** do gradiente da regressão linear — só que agora `p = σ(Xβ)`.

### Fronteira de decisão
$p = 0{,}5 \iff z = 0 \iff \beta_0 + \beta_1x_1 + \beta_2x_2 = 0$ → uma **reta**:
$$x_2 = -\frac{\beta_0 + \beta_1x_1}{\beta_2}$$
Nos dados da aula, após 1000 épocas: $\beta \approx [0{,}257;\ 2{,}116;\ 2{,}653]$ e $J \approx 0{,}186$.

---

## Exercício guiado — 6 passos ([`ExercicioGuiado.py`](ExercicioGuiado.py))

| # | Passo | Detalhe |
|---|---|---|
| 1 | Gerar e visualizar os dados | duas nuvens gaussianas em (−1,−1) e (1,1), 50 por classe, semente 42 |
| 2 | Separar treino/teste e montar `X` | 80/20 por permutação de índices; `X` = coluna de 1s + `x1` + `x2` → `(100, 3)` |
| 3 | `sigmoid`, `custo` e `gradiente` | `np.clip(p, 1e-12, 1-1e-12)` evita `log(0)` |
| 4 | Treinar e guardar o histórico | α = 0,1, 1000 épocas |
| 5 | Probabilidades, classes e acurácia | limiar 0,5 → acurácia **0,90** no treino e **0,90** no teste |
| 6 | Desenhar a fronteira e a curva de custo | reta `p = 0,5` sobre os pontos |

```bash
cd "Aula6/Aula 6.4"
python ExercicioGuiado.py
```

---

## Erros comuns
| # | Erro | Correção |
|---|---|---|
| 1 | Confundir com regressão linear (usar `z` como resposta, MSE como custo) | `p = sigmoid(X @ beta)` e entropia cruzada |
| 2 | Probabilidade tratada como classe | `predict` devolve `(predict_proba(X) >= 0.5).astype(int)` |
| 3 | Esquecer a sigmoide (copiar o laço da 6.3) | valores fora de (0, 1) geram `nan` no log |
| 4 | Sinal errado na atualização | `beta = beta - alpha * grad` (somando, o custo sobe a cada época) |
| 5 | Esquecer a coluna do intercepto | a fronteira fica presa à origem |
| 6 | Entropia cruzada incompleta | não esquecer o termo da classe 0 nem o sinal de menos |
| 7 | `log(0)` no custo | `np.clip(p, 1e-12, 1 - 1e-12)` |
| 8 | Divergência despercebida (só olhar o β final) | plotar o `historico`; com α = 100 o custo sobe em 483 das 1000 épocas |
| 9 | Dimensões incompatíveis | `y (80, 1)` com `p (80,)` vira `(80, 80)` sem erro — manter tudo `(m,)` |
| 10 | Acurácia sobre probabilidades | `np.mean((p >= 0.5) == y)`, não `np.mean(p == y)` |
| 11 | Fronteira incompatível com β | `x2 = -(b0 + b1 * x1) / b2` (dividir por `b2`) |
| 12 | Avaliar só no treino | o treino é otimista; reporte o teste |

## Desafio de casa — classe `RegressaoLogistica` (~40 linhas)

```python
class RegressaoLogistica:
    def __init__(self, alpha=0.1, epocas=1000): ...
    def fit(self, X, y): ...              # treina e guarda self.beta e self.historico
    def predict_proba(self, X): ...       # probabilidades da classe 1
    def predict(self, X, limiar=0.5): ... # classes 0/1
    def score(self, X, y): ...            # acurácia
```

**Critérios de aceite:** `len(historico)` igual ao número de épocas · primeiro custo = ln 2 ≈ **0,6931** (β começa em zero) · `predict_proba` em (0, 1) e `predict` só 0 e 1 · **acurácia no teste ≥ 0,85** · fronteira desenhada corretamente.

> O `LogisticRegression` do scikit-learn serve só para comparação: por padrão ele aplica regularização L2 (`C = 1.0`), então os coeficientes **não** coincidem exatamente com os nossos.

## Leitura
NumPy: `numpy.exp`, `numpy.clip`, `numpy.matmul` · Matplotlib: `pyplot.scatter`, `pyplot.plot` · scikit-learn: `LogisticRegression`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.4 Regressao Logistica.pdf` | slides da aula |
| `ExercicioGuiado.py` | os 6 passos do exercício guiado |

---

**← Anterior:** [Aula 6.3 — Gradiente descendente](../Aula%206.3/README.md) · **Próxima →** [Aula 6.5 — Métricas na mão](../Aula%206.5/README.md)
