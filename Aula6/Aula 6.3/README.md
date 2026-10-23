# Aula 6.3 — Gradiente descendente do zero

> **Módulo 06 · ML na mão, com NumPy** · 2 h
> Material: [`6.3 - Gradiente Descendente.pdf`](6.3%20-%20Gradiente%20Descendente.pdf) (36 slides) + [`Exercicio.py`](Exercicio.py)

A mesma regressão linear da Aula 6.2, mas agora **encontrada passo a passo** em vez de numa única conta. A equação normal precisa montar e resolver $X^TX$ — inviável com milhões de linhas ou com modelos que não têm fórmula fechada (como a regressão logística da 6.4 e as redes neurais). O gradiente descendente só precisa de uma coisa: saber **para que lado o custo sobe**, e dar um passo para o lado contrário.

> **Analogia:** descer uma montanha no escuro, com neblina. Você não vê o vale; só sente a inclinação do chão onde pisa. A **altura** é o erro; a **inclinação** é o gradiente; o **tamanho do passo** é a taxa de aprendizado α.

---

## Objetivos

1. Explicar o significado de $\beta_0$, $\beta_1$, $\alpha$, $J(\beta)$ e $\nabla J$, e onde cada um aparece no código.
2. Calcular **à mão** uma iteração completa do gradiente descendente.
3. Implementar o gradiente descendente vetorizado em NumPy, sem otimizador pronto.
4. Diagnosticar convergência e divergência pela curva de custo.
5. Animar a trajetória dos coeficientes sobre as curvas de nível do custo.

---

## Teoria

### Função de custo
$$J(\beta) = \frac{1}{2m}\sum_{i=1}^{m}(\hat y_i - y_i)^2$$
Erro individual, SSE, MSE e $J$ medem a mesma coisa em escalas diferentes (o ½ só simplifica a derivada).

### Gradiente e regra de atualização
$$\frac{\partial J}{\partial \beta_0} = \frac{1}{m}\sum(\hat y_i - y_i) \qquad \frac{\partial J}{\partial \beta_1} = \frac{1}{m}\sum(\hat y_i - y_i)\,x_i$$
$$\beta_j \leftarrow \beta_j - \alpha\,\frac{\partial J}{\partial \beta_j} \qquad\Longleftrightarrow\qquad \beta \leftarrow \beta - \alpha\nabla J,\quad \nabla J = \tfrac{1}{m}X^T(\hat y - y)$$

Derivada **negativa** → o coeficiente **aumenta**; positiva → diminui; ≈ 0 → quase não muda.

### Primeira iteração à mão
`x = [1, 2, 3]`, `y = [2, 4, 6]`, $\beta_0 = \beta_1 = 0$, $\alpha = 0{,}1$

| | t = 0 | t = 1 |
|---|---|---|
| $\beta_0$ | 0 | **0,4** |
| $\beta_1$ | 0 | **0,9333** |
| $J$ | 9,3333 | **1,8815** |

Após ~300 iterações: $\beta_0 \approx 0{,}02$, $\beta_1 \approx 1{,}99$ (reta ideal: $\hat y = 2x$).

### Taxa de aprendizado
- **Muito pequena:** converge, mas devagar.
- **Adequada:** desce rápido e estabiliza.
- **Muito alta:** oscila e **diverge** (o custo explode).

**Épocas e convergência:** pare quando $|J_{t-1} - J_t| <$ tolerância (ex.: `1e-8`).

### Batch × estocástico × mini-batch

| | Amostras por gradiente | Trajetória | Uso típico |
|---|---|---|---|
| **Batch** | todas (m) | suave, estável | bases pequenas — esta aula |
| **Estocástico (SGD)** | 1 | ruidosa, oscila perto do mínimo | fluxos de dados, aprendizado online |
| **Mini-batch** | b (ex.: 32) | levemente ruidosa | equilíbrio; padrão em deep learning |

---

## Exercício guiado ([`Exercicio.py`](Exercicio.py))

1. **Versão 1 — laço explícito:** atualiza `b0` e `b1` com as duas derivadas parciais.
2. **Versão 2 — vetorizada:** monta `X = [1s | x]`, usa `gradiente = X.T @ erro / m`, guarda `historico` do custo e `trajetoria` de β, e para por tolerância.
   *Conferência:* na 1ª passagem, custo **9,3333** e `beta = [0.4, 0.9333]`.
3. **Mapa do custo:** `np.meshgrid` + `plt.contour` com a trajetória por cima, e a superfície 3D (`plot_surface`).
4. **Animação:** `FuncAnimation` desenha a descida e salva `descida.gif`.

```bash
cd "Aula6/Aula 6.3"
python Exercicio.py
```
> O último trecho usa `IPython.display.Image` para mostrar o GIF — funciona no VS Code (modo interativo), Jupyter e Colab. Rodando como script, o GIF é salvo na pasta atual.

---

## Erros comuns
**Configuração e dimensões**

| Erro | Sintoma | Correção |
|---|---|---|
| Taxa de aprendizado alta (`alpha = 5.0`) | custo cresce, vira `inf` ou `nan` | `alpha = 0.1` e observe o custo |
| Épocas insuficientes (`epocas = 10`) | β longe do esperado, custo ainda caindo | mais épocas + critério de parada (`tol = 1e-8`) |
| Esquecer o intercepto (`X = x.reshape(-1, 1)`) | reta forçada a passar pela origem | `X = np.column_stack([np.ones(m), x])` |
| Dimensões incorretas (`y.reshape(-1, 1)`) | `y (m,1) − ŷ (m,)` vira matriz `(m, m)` | `y = y.ravel()` e `assert erro.shape == (m,)` |
| Não registrar o custo | impossível saber se aprendeu | `historico.append(custo)` a cada época |

**Sinais e escala**

| Erro | Sintoma | Correção |
|---|---|---|
| Sinal trocado (`beta + alpha * gradiente`) | custo cresce mesmo com α pequeno | `beta - alpha * gradiente` |
| Atualizar com o custo (`beta - alpha * custo`) | todos os β mudam igual, sem direção | usar o **gradiente** |
| Erro invertido (`erro = y - y_hat`) | gradiente aponta para cima | `erro = y_hat - y` |
| Escalas muito diferentes (área = 200, quartos = 3) | um único α não serve para as duas | padronizar as features antes

## Desafio de casa — classe `GradienteDescendente`

```python
class GradienteDescendente:
    def __init__(self, alpha=0.1, epocas=1000, tol=1e-8): ...  # guarda os hiperparâmetros
    def fit(self, X, y): ...      # adiciona 1s, treina beta_, salva historico_
    def predict(self, X): ...     # retorna ŷ
    def score(self, X, y): ...    # retorna o R²
```

**Critérios de aceite:** executa sem erros · coeficientes próximos dos esperados · custo diminui com um treino bem configurado · `predict` correto e `score` devolve o R² · a animação mostra a trajetória de β sobre as curvas de nível · você explica β, α, gradiente e custo com suas palavras.

## Leitura
NumPy: `numpy.matmul` (`@`), `ndarray.T`, `numpy.column_stack`, *Broadcasting*, `numpy.linalg.pinv` (para comparar) · Matplotlib: `Axes.contour`, `Axes3D.plot_surface`, `animation.FuncAnimation`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `6.3 - Gradiente Descendente.pdf` | slides da aula |
| `Exercicio.py` | versões com laço e vetorizada, mapa de custo, superfície 3D e animação |

---

**← Anterior:** [Aula 6.2 — Equação normal](../Aula%206.2/README.md) · **Próxima →** [Aula 6.4 — Regressão logística](../Aula%206.4/README.md)
