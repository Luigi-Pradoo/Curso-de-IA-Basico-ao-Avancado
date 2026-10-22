# Aula 5.3 — Pré-processamento com `Pipeline` e `ColumnTransformer`

> **Módulo 05 · Fundamentos práticos do ML** · 2 h · Dataset: **Titanic** (OpenML)
> Material: [`Aula 5.3.pdf`](Aula%205.3.pdf) (61 slides) + [`ExercicioGuiado.py`](ExercicioGuiado.py)

A Aula 5.2 deu a regra: *o que aprende algo dos dados aprende só com o treino*. Esta aula transforma a regra em **estrutura de código**. Imputação, codificação e escalonamento viram um objeto só — um `Pipeline` com caminhos separados para colunas numéricas e categóricas — que recebe o DataFrame cru, com nulos e texto, e devolve uma previsão. Como o objeto aprende tudo no `fit`, ele respeita a fronteira treino/teste **por construção**, não por disciplina do programador.

---

## Objetivos

1. Explicar o que o pré-processamento faz e por que ele respeita a fronteira treino/teste.
2. Dizer o que `fit`, `transform` e `fit_transform` fazem — e onde cada um entra.
3. Imputar ausentes, escalonar numéricas e codificar categóricas com one-hot.
4. Escrever um `Pipeline` que vá do DataFrame cru à predição, sem etapa manual.
5. Separar os caminhos numérico e categórico com `ColumnTransformer`.
6. Montar o pipeline completo do Titanic e avaliá-lo uma vez no teste trancado.

---

## Teoria

### Por que o scikit-learn recusa o dado cru
```
ValueError: could not convert string to float: 'male'
ValueError: Input X contains NaN.
```
Todo estimador clássico espera uma **matriz numérica, densa e completa**. O pré-processamento existe para produzir essa matriz — **do mesmo jeito** no treino, no teste e em produção.

### Três operações, uma ordem
1. **Imputação** — preencher o que falta (numérica: mediana/média; categórica: moda ou `"missing"`).
2. **Codificação** — categoria → número (`OneHotEncoder` sem ordem; `OrdinalEncoder` com ordem real). Só depois de imputar.
3. **Escalonamento** — numéricas na mesma régua. Obrigatório com distância/regularização; irrelevante em árvores. Sempre por último no caminho numérico.

### Uma API só, para tudo

| Método | Quem tem | O que faz |
|---|---|---|
| `fit(X, y)` | transformador e estimador | aprende estatísticas/parâmetros e guarda no objeto |
| `transform(X)` | transformador | aplica o que foi aprendido |
| `fit_transform(X)` | transformador | os dois de uma vez — **só no treino** |
| `predict(X)` | estimador | devolve a predição |

Atributos aprendidos terminam em `_` (`statistics_`, `mean_`, `scale_`, `categories_`). O `StandardScaler` guarda `mean_` e `scale_` e aplica $z = (x - \mu_{treino}) / \sigma_{treino}$ a qualquer linha que chegue depois.

### Escalonadores

| Transformador | Fórmula | Quando |
|---|---|---|
| `StandardScaler` | $(x-\mu)/\sigma$ | padrão em modelos lineares e SVM |
| `MinMaxScaler` | $(x-\min)/(\max-\min)$ | quando o intervalo [0,1] é exigido |
| `RobustScaler` | $(x-\text{mediana})/\text{IQR}$ | outliers pesados, como `fare` |

### `OneHotEncoder` — parâmetros que importam
- `handle_unknown="ignore"`: categoria nova no teste vira uma linha de zeros em vez de exceção.
- `drop="first"`: evita colinearidade perfeita em modelos lineares (não use em árvores).
- `min_frequency` / `max_categories`: agrupam categorias raras.

### `Pipeline` e `ColumnTransformer`

```text
PIPELINE — UM OBJETO SÓ
                    ┌─ numéricas:   SimpleImputer(median) → StandardScaler ─┐
DataFrame cru  ──►  │                                                        ├─► matriz densa ─► LogisticRegression
(com nulos)         └─ categóricas: SimpleImputer(moda)   → OneHotEncoder  ─┘
```

- `pipe.fit(X_tr, y_tr)` chama `fit_transform` em cada passo; `pipe.predict(X_te)` chama só `transform`. É essa assimetria que faz do Pipeline uma **medida de segurança**.
- `remainder="drop"` (padrão) descarta em silêncio as colunas não listadas → confira sempre `prep.fit_transform(X_tr).shape`.
- Dentro de `cross_val_score`, mediana, moda e desvio são **recalculados a cada fold**.
- `get_feature_names_out()` dá nome às colunas geradas — é o que dá sentido a `coef_` e, mais tarde, ao SHAP.

---

## Exercício guiado — Titanic em 14 passos ([`ExercicioGuiado.py`](ExercicioGuiado.py))

| # | Passo | Resultado de referência |
|---|---|---|
| 1 | Carregar 8 colunas, sem tocar em nada | `(1309, 8)` |
| 2 | **Separar primeiro** (`test_size=0.2, stratify=y, random_state=42`) | `X_te` não é lido até o passo 9 |
| 3 | Diagnosticar nulos **só no treino** | `age` 218 · `embarked` 2 · `fare` 1 |
| 4 | Declarar grupos de colunas + `assert` | num: `age, fare, sibsp, parch` · cat: `pclass, sex, embarked` |
| 5 | Caminho numérico | imputer + `StandardScaler` |
| 6 | Caminho categórico | imputer (moda) + `OneHotEncoder(handle_unknown="ignore")` |
| 7 | Juntar com `ColumnTransformer` | `(1047, 12)` — 4 numéricas + sex(2) + embarked(3) + pclass(3) |
| 8 | Modelo inteiro + `cross_val_score(cv=5)` | ≈ **0,789 ± 0,020** |
| 9 | Abrir o teste — **uma vez** | ≈ **0,794** (dentro de um desvio da validação) |
| 10 | Ler os coeficientes com nome | ser homem e 3ª classe pesam contra; `sex_female` e `pclass_1` a favor |
| 11 | `GridSearchCV` no pré-processamento | chaves `prep__num__imputer__strategy`, `clf__C`… |
| 12 | `FunctionTransformer(np.log1p)` em `fare` | transformação sem estado, mas mora no pipeline |
| 13 | Transformador próprio (`TamanhoFamilia`) | `BaseEstimator` + `TransformerMixin` |
| 14 | Salvar com `joblib` e prever uma linha crua (com `age=None`) | probabilidade ≈ 0,09 |

> Os números podem variar um pouco entre versões do scikit-learn; o que importa é a validação cruzada e o teste concordarem.

### Como rodar
```bash
cd "Aula5/Aula 5.3"
python ExercicioGuiado.py
```
Baixa o Titanic do OpenML na primeira execução (precisa de internet) e grava `titanic_pipeline.joblib` na pasta.

---

## Erros comuns

| Erro | Por que dói | Correção |
|---|---|---|
| `scaler.fit(X)` antes do split | média e desvio carregam o teste | split primeiro, sempre |
| `fit_transform(X_test)` | o teste reescreve os parâmetros do treino | só `transform` — ou deixe o pipeline chamar |
| `StandardScaler` em coluna categórica | média de rótulo não significa nada | categórica vai pelo caminho `cat` |
| One-hot em `fare` | uma coluna por preço: o modelo decora | one-hot é para categoria, não contínuo |
| Esquecer `handle_unknown="ignore"` | categoria nova derruba o `predict` | declarar no codificador e testar com linha inventada |
| Coluna esquecida nas listas | `remainder="drop"` descarta em silêncio | conferir `len(num) + len(cat)` contra `X.shape[1]` |

> Procure um `fit` **acima** da linha do `train_test_split`. Se existe, há vazamento.

## Desafio de casa

Feche o notebook da aula e construa o pipeline do Titanic **do zero**, do DataFrame cru até a métrica final. Se trocar a `LogisticRegression`, troque só a última linha do pipeline.

**Critérios de aceite:** colunas numéricas e categóricas listadas e justificadas · ausentes tratados nos dois caminhos · `StandardScaler` nas numéricas · `OneHotEncoder` com `handle_unknown` nas categóricas · tudo dentro de um `ColumnTransformer`, dentro de um `Pipeline` · split antes de qualquer `fit` · teste avaliado uma única vez · um parágrafo explicando por que o pipeline reduz inconsistências.

**Pergunta conceitual (vale metade da nota):** *por que não devemos calcular a mediana, a média ou a escala usando o dataset inteiro antes do split?* Responda pelo mecanismo: qual número atravessou a fronteira, o que ele carregava do teste e o que isso faz com a métrica.

## Leitura
`Pipeline` · `ColumnTransformer` · `SimpleImputer` · `StandardScaler` · `OneHotEncoder` · guia *Common pitfalls* — documentação do scikit-learn.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 5.3.pdf` | slides da aula |
| `ExercicioGuiado.py` | os 14 passos do exercício |
| `titanic_pipeline.joblib` | pipeline treinado salvo no passo 14 |

---

**← Anterior:** [Aula 5.2 — Treino, validação e teste](../Aula%205.2/README.md) · **Próxima →** [Aula 5.4 — Feature engineering](../Aula%205.4/README.md)
