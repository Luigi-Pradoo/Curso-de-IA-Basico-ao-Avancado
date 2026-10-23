# Aula 6.6 — Treino e validação cruzada implementada

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Dataset sintético desbalanceado (90 / 10)
> Material: [`Aula 6.6 - Validacao Cruzada.pdf`](Aula%206.6%20-%20Validacao%20Cruzada.pdf) (55 slides) + [`Exercicio Guiado.py`](Exercicio%20Guiado.py)

A pergunta da aula: **o número no fim mede o modelo, ou a divisão que sorteamos?** Mesmo modelo, mesmos dados, três divisões treino/teste diferentes → três acurácias diferentes. A validação cruzada responde a isso girando o papel de validação entre K partes do dado e resumindo os K resultados com **média e desvio-padrão**. Aqui escrevemos tudo do zero — `meu_kfold`, `meu_stratified_kfold` e `meu_cross_val_score` — e só no fim conferimos com o scikit-learn.

---

## Objetivos

1. Explicar por que uma única divisão treino/teste dá uma estimativa instável.
2. Construir os índices de treino e validação do K-Fold com NumPy.
3. Implementar um Stratified K-Fold que preserva a proporção das classes.
4. Escrever o seu próprio `cross_val_score`, com **um modelo novo por rodada**.
5. Resumir os scores com média e desvio-padrão, sem vazamento de dados.

---

## Teoria

### Vocabulário
- **K** — número de partes; também é o número de rodadas e de treinamentos.
- **Fold** — uma das K partes (um grupo de amostras).
- **Rodada** — treinar com K − 1 folds e validar no fold restante.

Cada amostra é validação **exatamente uma vez**. Quando `n` não é múltiplo de `K`, os primeiros `n % K` folds ganham uma amostra a mais (`n = 10, K = 3 → tamanhos [4, 3, 3]`).

### Ferramentas de NumPy/Python usadas
Índice × valor · selecionar linhas com arrays de índices (`X[train_idx]`) · **embaralhar os índices, não os dados** (`rng.permutation`) · fatiamento e `np.concatenate` · **funções geradoras com `yield`** (entregam um par, pausam e continuam).

### K-Fold × Stratified K-Fold
Com o dataset **ordenado por classe** e K-Fold sem embaralhar, as rodadas 1–4 validam só na classe 0 e a rodada 5 treina só com a classe 0. O **Stratified K-Fold** reparte os índices **de cada classe** entre os K folds, e cada fold herda a proporção do dataset.

| Use | Quando |
|---|---|
| K-Fold | regressão (y contínuo) ou classes balanceadas |
| Stratified K-Fold | classificação, sobretudo com classes desbalanceadas |

### Resumir os scores
$$\bar s = \frac{1}{K}\sum_{k=1}^{K} s_k \qquad \sigma = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(s_k - \bar s)^2}$$
(`np.std` usa o desvio **populacional**, `ddof=0`.) Dois modelos com a mesma média podem ter dispersões muito diferentes — o desvio diz quanto do número é sorte de divisão.

### Fábrica de modelos e vazamento
- Passe uma **fábrica** (a classe, sem parênteses) e crie `modelo = fabrica()` **dentro** do laço: cada rodada começa do zero.
- Padronização e qualquer estatística aprendida ficam **dentro do fold** (calculadas só com o treino daquela rodada).
- Fluxo completo: desenvolvimento (80%, com validação cruzada) → **teste final (20%), aberto uma vez**.
- **Quando K-Fold não basta:** séries temporais (embaralhar usa o futuro para prever o passado) e amostras agrupadas (o mesmo paciente/cliente nos dois lados).

---

## Exercício guiado — 12 etapas ([`Exercicio Guiado.py`](Exercicio%20Guiado.py))

Até a etapa 12, **sem** `KFold`, `StratifiedKFold` ou `cross_val_score` do sklearn.

| Bloco | Etapas |
|---|---|
| **Dados** | 1 criar o dataset (90 pontos da classe 0 em (0,0), 10 da classe 1 em (2,2)) · 2 identificar `X (100, 2)` e `y (100,)` · 3 distribuição das classes |
| **Folds** | 4 `meu_kfold` (com verificação: cada índice é validação uma vez; treino e validação não se cruzam) · 6 `meu_stratified_kfold` · 5 visualizar os folds das três estratégias · 7 classes por fold com `np.bincount` |
| **Avaliação** | 8 modelo de referência `CentroideMaisProximo` (fábrica: `m1 is m2 → False`) · 9 accuracy por fold, laço escrito à mão · 10 média e desvio (manual × `np.mean`/`np.std`) · 11 `meu_cross_val_score` com as três formas de dividir + baseline "sempre 0" (0,90) |
| **Comparação** | 12 conferir com o scikit-learn: `meu_kfold == KFold` sem shuffle; nossos folds no `cross_val_score` oficial; `StratifiedKFold` do sklearn; e o caso em que o `KFold` sem shuffle faz um fold treinar só com a classe 0 |

```bash
cd "Aula6/Aula 6.6"
python "Exercicio Guiado.py"
```

---

## Erros comuns

**Índices e folds**

| Situação | Problema | Correção |
|---|---|---|
| Ler K = 5 como "5 amostras por fold" | K é o número de folds → rodadas e tamanhos errados | tamanho = `n // K` |
| `metrica(val_idx, y_pred)` | índice é posição, não rótulo → métrica sem sentido, sem erro | `metrica(y[val_idx], y_pred)` |
| `modelo.fit(X[val_idx], …)` | papéis trocados: treina com 20, avalia em 80 | conferir os tamanhos de cada rodada |
| `indices[fim - 1:]` | fatias sobrepostas ou com buraco | `intersect1d` vazio; união = todos |
| `rng.shuffle(X); rng.shuffle(y)` | duas permutações diferentes → rótulos desalinhados | permutar **índices** uma vez |

**Modelo e dados**

| Situação | Problema | Correção |
|---|---|---|
| `modelo.fit(X, y)` antes do laço | o modelo já viu as validações | `fit` dentro do laço, com `X[train_idx]` |
| `modelo = Modelo()` antes do `for` | o estado da rodada anterior persiste | `modelo_factory()` dentro do `for` |
| `shuffle=True` sem semente | folds mudam a cada execução | fixar `random_state` |
| K-Fold comum em base 90 / 10 | folds sem a classe minoritária | Stratified K-Fold e métricas por classe |

**Avaliação**

| Situação | Problema | Correção |
|---|---|---|
| Padronizar `X` inteiro antes da CV | a validação influencia a transformação → score otimista | padronizar dentro do fold |
| Escolher hiperparâmetros pelo teste | o teste vira parte da seleção | escolher pela CV, testar uma vez |
| "Em produção o modelo terá 0,89" | a média **estima**, não garante | reportar média ± desvio |
| `shuffle=True` em vendas diárias | treina com o futuro | divisão temporal |

## Desafio de casa — construa seu próprio `cross_val_score`
Dataset: `load_breast_cancer` (569 amostras, 212 / 357).
1. **K-Fold** — gerador de pares de índices de treino e validação, com shuffle e semente.
2. **Stratified K-Fold** — gerador que preserva aproximadamente a proporção das classes.
3. **Cross-val-score** — modelo novo por rodada, métrica, scores, média e desvio.

**Entregar:** índices dos folds · distribuição de classes · scores individuais · média e desvio · gráfico dos scores · comparação com o scikit-learn.
**Critérios de aceite:** cada amostra é validação uma vez no K-Fold · treino e validação sem índices em comum na mesma rodada · o Stratified preserva aproximadamente as proporções · cada rodada usa um modelo novo · métricas corretas · média e desvio corretos, com o `ddof` declarado · resultados reprodutíveis com semente · código organizado e comentado · você explica o funcionamento das funções · comparação com o scikit-learn sob divisões equivalentes.

## Leitura
NumPy: `arange`, `Generator.permutation`, `unique`, `array_split` · Python: *generator* · scikit-learn: `KFold`, `StratifiedKFold`, `cross_val_score`.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.6 - Validacao Cruzada.pdf` | slides da aula |
| `Exercicio Guiado.py` | as 12 etapas, de `meu_kfold` à comparação com o sklearn |

---

**← Anterior:** [Aula 6.5 — Métricas na mão](../Aula%206.5/README.md) · **Próxima →** [Aula 6.7 — KNN do zero](../Aula%206.7/README.md)
