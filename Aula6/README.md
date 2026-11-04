# Aula 6 — Machine Learning na mão, com NumPy (Módulo 06)

> **Escolinha de IA · Módulo 06** · 13 aulas de 2 h (9 publicadas aqui) · NumPy puro
> Nenhum `import` de sklearn para treinar: cada algoritmo é derivado no quadro, escrito em NumPy e **só então** comparado com a biblioteca.

Em todo o Módulo 05 o treino coube numa linha: `model.fit(X_train, y_train)`. Dentro dela existem uma matriz `X`, um vetor de pesos `w` e um gradiente que diz como ajustar `w`. Este módulo abre essa caixa. Ao final, cada aluno tem a **própria mini-biblioteca de ML no GitHub** — e a versão do scikit-learn passa a ser óbvia.

## Aulas do módulo

| Aula | Tema | O que você implementa |
|---|---|---|
| [6.1](Aula%206.1/README.md) | Álgebra linear e cálculo mínimos para ML | `algebra.py`: normas, distâncias, MSE e gradiente vetorizados |
| [6.2](Aula%206.2/README.md) | Regressão linear pela equação normal | classe `MinimosQuadrados` (inv, solve, pinv) |
| [6.3](Aula%206.3/README.md) | Gradiente descendente do zero | classe `GradienteDescendente` + animação da descida |
| [6.4](Aula%206.4/README.md) | Regressão logística do zero | classe `RegressaoLogistica` |
| [6.5](Aula%206.5/README.md) | Métricas implementadas na mão | `meu_classification_report`, ROC e AUC |
| [6.6](Aula%206.6/README.md) | Validação cruzada implementada | `meu_kfold`, `meu_stratified_kfold`, `meu_cross_val_score` |
| [6.7](Aula%206.7/README.md) | KNN do zero | classe `KNNDoZero` (loops × broadcasting) |
| [6.8](Aula%206.8/README.md) | Naive Bayes do zero | classe `NaiveBayesDoZero` (filtro de spam) |
| [6.9](Aula%206.9/README.md) | Árvore de Decisão do zero | classe `ArvoreDecisaoDoZero` (Gini, melhor corte, recursão) + 🌳 [laboratório interativo](Aula%206.9/arvore-interativa.html) |

## Como as aulas se encaixam

```text
6.1 álgebra + gradiente ──► 6.2 equação normal (resolve ∇J = 0 numa conta)
                         └─► 6.3 gradiente descendente (anda contra ∇J) ──► 6.4 regressão logística
6.5 métricas ──► 6.6 validação cruzada ──► 6.7 KNN (escolher K por CV) ──► 6.8 Naive Bayes
                                                                        └─► 6.9 Árvore de Decisão (max_depth × overfitting)
```

## Formato de cada aula

`01 Objetivos → 02 Recapitulação → 03 Teoria → 04 Quiz → 05 Exercício guiado → 06 Erros comuns → 07 Desafio de casa → 08 Leitura` — o desafio é corrigido no início da aula seguinte.

## Ambiente

```bash
pip install numpy matplotlib scikit-learn
```

O scikit-learn aparece só para **conferir** os resultados (e para carregar datasets).

---

**← Anterior:** [Aula 5 — Fundamentos práticos do ML](../Aula5/README.md)
