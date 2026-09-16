# ============================================================
# ATIVIDADE 1 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# DATASET IRIS
# ============================================================

# ============================================================
# 1. IMPORTANDO AS BIBLIOTECAS
# ============================================================

from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. CARREGANDO O DATASET IRIS
# ============================================================

iris = load_iris(as_frame=True)

df_iris = iris.frame

# Criando uma coluna com o nome da espécie
df_iris["species"] = df_iris["target"].map(
    dict(enumerate(iris.target_names))
)

print("\n========== DATASET IRIS ==========\n")
print(df_iris)


# ============================================================
# 3. HEAD()
# ============================================================

print("\n========== PRIMEIRAS LINHAS ==========\n")

print(df_iris.head())


# ============================================================
# 4. SHAPE
# ============================================================

print("\n========== DIMENSÕES ==========\n")

print(df_iris.shape)


# ============================================================
# 5. COLUMNS
# ============================================================

print("\n========== COLUNAS ==========\n")

print(df_iris.columns)


# ============================================================
# 6. INFO()
# ============================================================

print("\n========== INFORMAÇÕES DO DATAFRAME ==========\n")

df_iris.info()


# ============================================================
# 7. VALORES AUSENTES
# ============================================================

print("\n========== VALORES AUSENTES ==========\n")

print(df_iris.isnull().sum())


# ============================================================
# 8. DUPLICATAS
# ============================================================

print("\n========== DUPLICATAS ==========\n")

print("Quantidade de duplicatas:", df_iris.duplicated().sum())


# ============================================================
# 9. DESCRIBE()
# ============================================================

print("\n========== ESTATÍSTICAS DESCRITIVAS ==========\n")

print(df_iris.describe())


# ============================================================
# 10. VALUE_COUNTS()
# ============================================================

print("\n========== QUANTIDADE DE CADA ESPÉCIE ==========\n")

print(df_iris["species"].value_counts())


# ============================================================
# 11. GROUPBY()
# ============================================================

print("\n========== MÉDIA DO COMPRIMENTO DA SÉPALA POR ESPÉCIE ==========\n")

media_sepalas = df_iris.groupby("species")["sepal length (cm)"].mean()

print(media_sepalas)


# ============================================================
# GROUPBY() — OUTRAS ESTATÍSTICAS
# ============================================================

print("\n========== ESTATÍSTICAS POR ESPÉCIE ==========\n")

estatisticas = df_iris.groupby("species")["sepal length (cm)"].agg(
    ["mean", "min", "max"]
)

print(estatisticas)


# ============================================================
# 12. GRÁFICO DE BARRAS
# ============================================================

plt.figure()

df_iris["species"].value_counts().plot(kind="bar")

plt.xlabel("Espécie")
plt.ylabel("Quantidade")
plt.title("Quantidade de exemplos por espécie")

plt.show()


# ============================================================
# 13. HISTOGRAMA
# ============================================================

plt.figure()

plt.hist(df_iris["sepal length (cm)"])

plt.xlabel("Comprimento da sépala (cm)")
plt.ylabel("Quantidade")
plt.title("Distribuição do comprimento da sépala")

plt.show()


# ============================================================
# 14. GRÁFICO DE DISPERSÃO
# ============================================================

plt.figure()

plt.scatter(
    df_iris["sepal length (cm)"],
    df_iris["petal length (cm)"]
)

plt.xlabel("Comprimento da sépala (cm)")
plt.ylabel("Comprimento da pétala (cm)")
plt.title("Comprimento da sépala x Comprimento da pétala")

plt.show()


# ============================================================
# 15. BOXPLOT
# ============================================================

plt.figure()

plt.boxplot(df_iris["sepal length (cm)"])

plt.ylabel("Comprimento da sépala (cm)")
plt.title("Distribuição do comprimento da sépala")

plt.show()


# ============================================================
# 16. CORRELAÇÃO
# ============================================================

print("\n========== CORRELAÇÃO ==========\n")

colunas_numericas = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

correlacao = df_iris[colunas_numericas].corr()

print(correlacao)


# ============================================================
# 17. INSIGHTS
# ============================================================

print("\n========== INSIGHTS ==========\n")

print("1. As três espécies possuem a mesma quantidade de exemplos no dataset.")

print("2. A espécie Virginica apresenta o maior comprimento médio de sépala.")

print("3. Existe uma relação positiva entre o comprimento da sépala e o comprimento da pétala.")

print("4. As variáveis relacionadas às pétalas apresentam forte correlação entre si.")