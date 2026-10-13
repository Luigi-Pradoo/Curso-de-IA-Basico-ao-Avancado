
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Carregar o dataset
dados = fetch_california_housing()

# Separar características e variável-alvo
X, y = dados.data, dados.target

# Conferir as dimensões
print(X.shape, y.shape)

# Conferir os nomes das características
print(dados.feature_names)

n = X.shape[0]  # Número de amostras
Xb = np.column_stack((np.ones(n), X))  # Adicionar uma coluna de 1s para o intercepto

print(Xb.shape)  # Conferir as dimensões após adicionar a coluna de 1s
print(Xb[0,:3])  # Conferir as primeiras 3 colunas da primeira linha do novo array

A = Xb.T @ Xb  # Matriz A = X^T * X
b = Xb.T @ y  # Vetor b = X^T * y

beta_inv = np.linalg.inv(A)  # Inversa da matriz A
beta_solve = np.linalg.solve(A, b)  # Resolver o sistema linear A * beta = b
beta_pinv = np.linalg.pinv(Xb) @ y  # Usando a pseudo-inversa para calcular beta

print (np.allclose(beta_solve, beta_pinv))  # Conferir se A^-1 * b é igual a beta_solve
print (np.abs(beta_inv - beta_pinv).max())  # Conferir a diferença máxima entre beta_inv e beta_pinv

y_hat = Xb @ beta_pinv  # Previsões do modelo usando beta_pinv

r2 = 1 - np.sum((y - y_hat)**2)/ np.sum((y - y.mean())**2)  # Calcular o R² do modelo
print (r2)  # Conferir o valor do R²

lr = LinearRegression()  # Criar o modelo de regressão linear
lr.fit(X, y)  # Treinar o modelo com os dados
beta_sk = np.r_[lr.intercept_,lr.coef_]  # Obter os coeficientes do modelo treinado

print (np.allclose(beta_pinv, beta_sk))  # Conferir se os coef
print (lr.score(X, y))  # Conferir o R² do modelo treinado