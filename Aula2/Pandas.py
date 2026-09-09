import pandas as pd

# =========================
# SERIES
# =========================

notas = pd.Series([8, 7, 9, 6])

print(notas)


# =========================
# CRIANDO UM DATAFRAME
# =========================

dados = {
    "Nome": ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"],
    "Idade": [20, 21, 19, 22, 23, 24],
    "Nota": [8, 7, 9, 6, 5, 10]
}

df = pd.DataFrame(dados)

print(df)


# =========================
# EXPLORANDO O DATAFRAME
# =========================

df.head()       # Primeiras linhas
df.tail()       # Últimas linhas
df.shape        # Quantidade de linhas e colunas
df.columns      # Nomes das colunas
df.info()       # Informações sobre o DataFrame
df.describe()   # Estatísticas descritivas


# =========================
# SELECIONANDO DADOS
# =========================

df["Nota"]              # Seleciona uma coluna
df[["Nome", "Nota"]]    # Seleciona várias colunas


# =========================
# FILTRANDO DADOS
# =========================

df[df["Nota"] >= 7]


# =========================
# ORDENANDO DADOS
# =========================

df.sort_values("Nota")                  # Menor para maior
df.sort_values("Nota", ascending=False) # Maior para menor


# =========================
# VALORES AUSENTES
# =========================

df.isnull().sum()