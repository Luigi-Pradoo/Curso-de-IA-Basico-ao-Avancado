import pandas as pd

Escolhinha_de_IA = {
    "Nome": ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"],
    "Idade": [20, 21, 19, 22, 23, 24],
    "Nota": [8, 7, 9, 6, 5, 10]
}

df = pd.DataFrame(Escolhinha_de_IA)

print(df.head(5))
print(df.tail(5))
print(df.shape)
print(df.columns)
print(df.info(5))
print(df.describe())
print(df["Nota"])
print(df[["Nome", "Nota"]])
print(df[df["Nota"] >= 7])
print(df.sort_values("Nota"))
print(df.sort_values("Nota", ascending=False))
print(df.isnull().sum())