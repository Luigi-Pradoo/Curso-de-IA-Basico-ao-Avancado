# Aula 4 — Análise Exploratória de Dados (EDA) com o Iris

> **Curso preparatório · Aula 4** · Conhecer o dado antes de treinar qualquer modelo
> Material: [`Exercicio.py`](Exercicio.py) (esta aula não tem PDF na pasta)
> Dataset: **Iris** — 150 flores, 4 medidas em cm, 3 espécies (50 de cada)

Todo projeto de ML começa olhando para o dado. Antes de perguntar "qual modelo usar?", a EDA responde: **quantas linhas e colunas existem, de que tipo, se há buracos ou duplicatas, como cada variável se distribui e quais variáveis andam juntas**. Esta aula aplica as ferramentas da Aula 3 (pandas + Matplotlib) num dataset real e termina com *insights* escritos — o mesmo Iris que a Aula 5.1 vai usar para treinar o primeiro modelo.

---

## Objetivos

1. Carregar um dataset do scikit-learn como DataFrame.
2. Fazer o "check-up" do dado: dimensões, tipos, nulos e duplicatas.
3. Resumir variáveis com `describe`, `value_counts` e `groupby`.
4. Visualizar distribuições e relações (barras, histograma, dispersão, boxplot).
5. Medir relação linear entre variáveis com a **correlação**.
6. Transformar números em conclusões escritas (*insights*).

---

## Roteiro do exercício (17 passos)

| # | Etapa | Ferramenta | Pergunta que responde |
|---|---|---|---|
| 1–2 | Importar e carregar | `load_iris(as_frame=True)` | — |
| 3 | Primeiras linhas | `head()` | como é uma linha do dado? |
| 4–5 | Dimensões e colunas | `shape`, `columns` | quantas amostras e features? |
| 6 | Tipos | `info()` | cada coluna é número ou texto? |
| 7 | Valores ausentes | `isnull().sum()` | falta algum dado? |
| 8 | Duplicatas | `duplicated().sum()` | há linhas repetidas? |
| 9 | Estatísticas | `describe()` | média, desvio, mínimo, quartis |
| 10 | Contagem por classe | `value_counts()` | as classes estão balanceadas? |
| 11 | Agregação | `groupby("species").agg(["mean","min","max"])` | as espécies diferem nas medidas? |
| 12 | Gráfico de barras | `.plot(kind="bar")` | visual do balanceamento |
| 13 | Histograma | `plt.hist` | como uma variável se distribui? |
| 14 | Dispersão | `plt.scatter` | duas variáveis andam juntas? |
| 15 | Boxplot | `plt.boxplot` | mediana, quartis e outliers |
| 16 | Correlação | `df.corr()` | força da relação linear (−1 a 1) |
| 17 | Insights | texto | o que aprendemos? |

### Insights esperados
1. As três espécies têm a **mesma quantidade** de exemplos (dataset balanceado).
2. *Virginica* tem o **maior comprimento médio de sépala**.
3. Há relação **positiva** entre comprimento da sépala e comprimento da pétala.
4. As medidas da **pétala** são fortemente correlacionadas entre si — e são as que melhor separam as espécies (isso volta na Aula 5.1).

> ⚠️ Repare em `duplicated().sum()`: o Iris tem uma linha repetida. Duplicatas parecem inofensivas na EDA, mas na Aula 5.2 viram uma das cinco formas de **vazamento de dados**.

---

## Como rodar

```bash
pip install pandas matplotlib scikit-learn
cd Aula4
python Exercicio.py
```

Cada gráfico abre numa janela; feche-a para o script continuar.

---

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Exercicio.py` | EDA completa do Iris, comentada passo a passo |

---

**← Anterior:** [Aula 3 — NumPy, pandas e Matplotlib](../Aula3/README.md) · **Próxima →** [Aula 5 — Fundamentos práticos do ML](../Aula5/README.md)
