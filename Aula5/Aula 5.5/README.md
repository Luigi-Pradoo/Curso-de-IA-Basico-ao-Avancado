# Aula 5.5 — Seleção de features: filtros, wrappers e embedded

> **Módulo 05 · Fundamentos práticos do ML** · 2 h · Dataset: **California Housing** + 8 colunas de ruído
> Material: [`Aula 5.5 - Seleção de Features.pdf`](Aula%205.5%20-%20Sele%C3%A7%C3%A3o%20de%20Features.pdf) (32 slides) + [`Exercicio.py`](Exercicio.py)

A Aula 5.4 respondeu *quais colunas criar*. Esta responde **quais colunas ficam**. Para tornar a pergunta mensurável, o exercício adiciona **8 colunas de puro ruído** às 8 reais do California Housing: um bom método de seleção tem que mandá-las embora. O critério da prática é concreto — **menos colunas, mesmo R² na validação cruzada** — e a lição técnica é a mesma do módulo inteiro: a seleção é um passo de treino, então mora **dentro do Pipeline**.

---

## Objetivos

1. Classificar qualquer método de seleção como **filtro**, **wrapper** ou **embedded**.
2. Ranquear features com ANOVA F e informação mútua e explicar por que os rankings divergem.
3. Rodar RFE, RFECV e `SequentialFeatureSelector` e estimar quantos ajustes cada um custa.
4. Usar Lasso e Random Forest (MDI e permutação) como seletores via `SelectFromModel`.
5. Colocar a seleção dentro do Pipeline, para que o score da validação cruzada seja honesto.

---

## Teoria

### Por que remover colunas
**Generalização** (ruído dá chance de decorar coincidências) · **Custo** (treino, inferência e coleta em produção) · **Interpretação** (seis colunas cabem numa explicação; sessenta não) · **Robustez** (cada coluna é uma dependência que pode quebrar).

### As três famílias — o critério é *quando o modelo participa*

| Família | Como funciona | Exemplos |
|---|---|---|
| **Filtro** | estatística por coluna; o modelo não participa | variância, correlação, ANOVA F, informação mútua |
| **Wrapper** | treina, pontua, repete sobre subconjuntos | RFE, RFECV, `SequentialFeatureSelector` |
| **Embedded** | a seleção sai de um único treino | Lasso (L1), importâncias de árvores |

### Filtros
- **Variância quase zero** — remova antes de qualquer outro método (depende da escala).
- **ANOVA F** (`f_regression` / `f_classif`) — mede relação **linear** com o alvo.
- **Informação mútua** — capta qualquer dependência, inclusive não linear; estimada por k-NN, estocástica, só serve para ranquear.
- **Limite dos filtros univariados:** `Latitude` e `Longitude` parecem fracas sozinhas (Pearson −0,14 e −0,05), mas juntas definem um ponto no mapa e valem muito.

### Wrappers
Com *p* colunas existem 2^p subconjuntos (p = 30 → mais de 1 bilhão). Por isso usamos buscas gulosas:
- **Forward** — começa vazio e adiciona a melhor coluna a cada rodada.
- **Backward / RFE** — começa com todas e remove a menos importante (`coef_` ou `feature_importances_`). **Escale antes** de usar `coef_`.
- **RFECV** — roda o RFE dentro de cada fold e escolhe o número de colunas.

### Embedded
- **Lasso** — a penalidade L1 zera coeficientes (a região L1 é um losango com quinas nos eixos). Com colunas correlacionadas, fica com uma quase ao acaso.
- **Árvores** — **MDI** (`feature_importances_`) é grátis, mas medida no treino e favorece ruído contínuo; **importância por permutação** é medida na validação e é mais confiável.

### Custo computacional (p = 20, CV 5 folds)
Filtros: uma passada · Lasso/RF: 1 treino · LassoCV com 100 alphas: ~500 ajustes · wrappers: centenas de ajustes (ex.: forward com 30 colunas escolhendo 3 → (30+29+28)×5 = **435 ajustes**).

### A seleção fica dentro do Pipeline

```python
# ❌ VAZA: o seletor viu as linhas de validação de todos os folds
X_sel = SelectKBest(f_regression, k=6).fit_transform(X_tr, y_tr)
cross_val_score(modelo, X_sel, y_tr, cv=cv)

# ✅ HONESTO: a seleção é refeita dentro de cada fold
pipe = Pipeline([("sel", SelectKBest(f_regression, k=6)), ("m", modelo)])
cross_val_score(pipe, X_tr, y_tr, cv=cv)
```

---

## Exercício guiado — 5 passos ([`Exercicio.py`](Exercicio.py))

Modelo fixo: `HistGradientBoostingRegressor`, `KFold(5, shuffle=True, random_state=42)`, métrica R².

| # | Passo | O que faz |
|---|---|---|
| 1 | Dados, ruído, teste trancado e baseline | 8 colunas reais + 8 `ruido_i`; baseline com as 16 colunas; faixa aceitável = R² ≥ baseline − 1 desvio |
| 2 | Filtros: ANOVA F e informação mútua | curva R² × k para k ∈ {2, 4, 6, 8, 12, 16}; menor k dentro da faixa; quais ruídos entraram |
| 3 | Wrapper com RFECV | `StandardScaler → RFECV(Ridge) → modelo` (validação aninhada) e gráfico do score por nº de colunas |
| 4 | Embedded com Lasso e Random Forest | `SelectFromModel(LassoCV)` (com e sem limiar `0.1*mean`); MDI × permutação lado a lado, ruído em cinza |
| 5 | Comparar e abrir o teste uma vez | tabela colunas / R² CV / desvio / tempo; escolhe o pipeline com **menos colunas dentro de 1 desvio** e só então avalia no teste |

### Como rodar
```bash
cd "Aula5/Aula 5.5"
python Exercicio.py
```
Baixa o California Housing na primeira execução. Os wrappers levam alguns minutos.

---

## Erros comuns
1. **Selecionar fora da validação** — o seletor vê os folds de validação e o score infla.
2. **Coeficientes sem escalar** — Lasso e RFE linear passam a ranquear pela unidade da coluna.
3. **Confiar só no MDI** — ruído contínuo recebe importância; confirme com permutação.
4. **Cortar "gêmeas" pela importância individual** — colunas correlacionadas dividem a importância.
5. **Escolher k olhando o teste.**
6. **Ler importância como causa** — importância diz que o modelo *usa* a coluna, não que ela *causa* o alvo.

**Para ir além:** stability selection, Boruta, mRMR e seleção por SHAP.

## Desafio de casa — "seis colunas, mesmo score"
Parta das 16 colunas da aula (8 reais + 8 ruído) mais as features que você criou na 5.4. Compare **um método de cada família** e entregue o melhor pipeline com **no máximo 6 colunas**.
**Critérios de aceite:** seleção dentro do Pipeline, validada com KFold de 5 folds e semente 42 · no máximo 6 colunas, com R² médio no máximo 0,01 abaixo do baseline · tabela com colunas, R², desvio e tempo para filtro, wrapper e embedded · nenhuma coluna de ruído no conjunto final · teste aberto uma única vez, na última célula.

## Leitura
- scikit-learn — Guia 1.13 *Feature selection* e Guia 4.2 *Permutation importance*
- `f_regression`, `mutual_info_regression`, `RFE`, `RFECV`, `SequentialFeatureSelector`, `SelectFromModel`, `LassoCV`, `fetch_california_housing`

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 5.5 - Seleção de Features.pdf` | slides da aula |
| `Exercicio.py` | exercício guiado completo (5 passos) |

---

**← Anterior:** [Aula 5.4 — Feature engineering](../Aula%205.4/README.md) · **Próxima →** [Aula 6 — ML na mão, com NumPy](../../Aula6/README.md)
