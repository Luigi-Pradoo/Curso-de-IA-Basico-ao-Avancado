# Aula 3 — NumPy, pandas e Matplotlib

> **Curso preparatório · Aula 3** · As três bibliotecas que aparecem em todo projeto de dados
> Material: 4 scripts de exemplo (esta aula não tem PDF na pasta)

Na Aula 2 a turma era uma lista de nomes e notas. Aqui ela vira **dado de verdade**: um array numérico (NumPy), uma tabela com colunas nomeadas (pandas) e um gráfico (Matplotlib). São essas três peças que carregam, limpam e descrevem um dataset — e que viram a primeira etapa do pipeline de Machine Learning a partir do Módulo 5.

---

## Objetivos

1. Criar arrays NumPy e fazer contas **vetorizadas** (sem `for`).
2. Montar um `DataFrame` do pandas e explorá-lo com `head`, `info`, `describe`, `shape`.
3. Selecionar colunas, **filtrar linhas com condição booleana** e ordenar.
4. Verificar valores ausentes com `isnull().sum()`.
5. Fazer os quatro gráficos básicos: barras, linhas, histograma e dispersão.

---

## Conteúdo

### NumPy — [`Numpy.py`](Numpy.py)
O array é a estrutura numérica de tudo em ML: uma operação vale para todos os elementos de uma vez.

```python
numeros = np.array([10, 20, 30, 40, 50])
numeros * 2            # [ 20  40  60  80 100]  ← sem laço
np.sum(numeros), np.mean(numeros), np.max(numeros), np.min(numeros)

matriz = np.array([[1, 2, 3], [4, 5, 6]])
matriz.shape           # (2, 3) → (linhas, colunas)
```

> Guarde o `shape`: no Módulo 5 o dataset é sempre `X.shape == (amostras, features)`.

### pandas — [`Pandas.py`](Pandas.py)

| Operação | Código |
|---|---|
| Série (uma coluna) | `pd.Series([8, 7, 9, 6])` |
| Tabela a partir de dicionário | `df = pd.DataFrame(dados)` |
| Primeiras / últimas linhas | `df.head()`, `df.tail()` |
| Dimensões e colunas | `df.shape`, `df.columns` |
| Tipos e nulos | `df.info()` |
| Estatísticas | `df.describe()` |
| Uma coluna / várias | `df["Nota"]`, `df[["Nome", "Nota"]]` |
| **Filtro booleano** | `df[df["Nota"] >= 7]` |
| Ordenar | `df.sort_values("Nota", ascending=False)` |
| Valores ausentes | `df.isnull().sum()` |

### Matplotlib — [`Matplotlib.py`](Matplotlib.py)

| Gráfico | Função | Quando usar |
|---|---|---|
| Barras | `plt.bar(nomes, faltas)` | comparar categorias |
| Linhas | `plt.plot(x, y)` | evolução / sequência |
| Histograma | `plt.hist(faltas)` | ver a **distribuição** de uma variável |
| Dispersão | `plt.scatter(aulas, faltas)` | relação entre duas variáveis numéricas |

Sempre com `xlabel`, `ylabel` e `title` — gráfico sem rótulo não comunica.

---

## Exercício — [`ExercicioPandas.py`](ExercicioPandas.py)

Com o DataFrame da Escolinha (nome, idade, nota), aplique em sequência: `head`, `tail`, `shape`, `columns`, `info`, `describe`, seleção de colunas, filtro de aprovados (`Nota >= 7`), ordenação crescente e decrescente e contagem de nulos.

---

## Como rodar

```bash
pip install numpy pandas matplotlib
cd Aula3
python Pandas.py
python Matplotlib.py     # abre uma janela por gráfico
```

---

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Numpy.py` | arrays, operações vetorizadas e `shape` |
| `Pandas.py` | Series, DataFrame, exploração, filtro e ordenação |
| `Matplotlib.py` | barras, linhas, histograma e dispersão |
| `ExercicioPandas.py` | exercício de exploração com pandas |

---

**← Anterior:** [Aula 2 — Python para IA](../Aula2/README.md) · **Próxima →** [Aula 4 — Análise Exploratória (EDA)](../Aula4/README.md)
