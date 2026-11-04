
import numpy as np
import time

# =====================================================
# 1. CALCULAR NORMA L1 OU L2
# =====================================================

def norma(x, ord=2):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor")

    if ord == 1:
        return np.sum(np.abs(x))

    if ord == 2:
        return np.sqrt(np.sum(x ** 2))

    raise ValueError("ord deve ser 1 ou 2")


# =====================================================
# 2. CALCULAR DISTÂNCIAS EUCLIDIANAS
# =====================================================

def distancias(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("Dimensões incompatíveis")

    norma_A2 = np.sum(A ** 2, axis=1, keepdims=True)
    norma_B2 = np.sum(B ** 2, axis=1)[None, :]

    dist2 = norma_A2 + norma_B2 - 2 * (A @ B.T)

    return np.sqrt(np.maximum(dist2, 0.0))


# =====================================================
# 3. CALCULAR ERRO QUADRÁTICO MÉDIO (MSE)
# =====================================================

def mse(X, y, w):
    erro = X @ w - y

    return np.mean(erro ** 2)


# =====================================================
# 4. CALCULAR GRADIENTE DO MSE
# =====================================================

def grad_mse(X, y, w):
    n = X.shape[0]
    erro = X @ w - y

    return (2 / n) * (X.T @ erro)


# =====================================================
# 5. CALCULAR GRADIENTE NUMÉRICO
# =====================================================

def grad_numerico(f, w, h=1e-6):
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)

    for j in range(w.size):
        passo = np.zeros_like(w)
        passo[j] = h

        grad[j] = (
            f(w + passo) - f(w - passo)
        ) / (2 * h)

    return grad


# =====================================================
# TESTES - EXECUÇÃO DIRETA
# =====================================================

print("\n========== TESTE 1: NORMAS ==========")

x = np.array([3, 4])

print("Vetor:", x)
print("Norma L1:", norma(x, ord=1))
print("Norma L2:", norma(x, ord=2))

print("Comparação L1:",
      np.allclose(norma(x, 1), np.linalg.norm(x, ord=1)))

print("Comparação L2:",
      np.allclose(norma(x, 2), np.linalg.norm(x, ord=2)))


# =====================================================
# TESTE 2: DISTÂNCIAS EUCLIDIANAS
# =====================================================

print("\n========== TESTE 2: DISTÂNCIAS ==========")

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print("Matriz A:")
print(A)

print("Matriz B:")
print(B)

print("Matriz de distâncias:")
print(distancias(A, B))

# Comparação com SciPy, se estiver instalado
try:
    from scipy.spatial.distance import cdist

    print("Comparação com scipy.cdist:",
          np.allclose(distancias(A, B), cdist(A, B)))

except ImportError:
    print("SciPy não instalado: comparação ignorada.")


# =====================================================
# TESTE 3: ERRO QUADRÁTICO MÉDIO
# =====================================================

print("\n========== TESTE 3: MSE ==========")

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])

y = np.array([1, 2, 3])
w = np.array([0.0, 0.0])

print("Matriz X:")
print(X)

print("Valores reais:", y)
print("Pesos:", w)

print("MSE:", mse(X, y, w))


# =====================================================
# TESTE 4: GRADIENTE DO MSE
# =====================================================

print("\n========== TESTE 4: GRADIENTE MSE ==========")

gradiente = grad_mse(X, y, w)

print("Gradiente analítico:", gradiente)


# =====================================================
# TESTE 5: GRADIENTE NUMÉRICO
# =====================================================

print("\n========== TESTE 5: GRADIENTE NUMÉRICO ==========")

f = lambda pesos: mse(X, y, pesos)

gradiente_num = grad_numerico(f, w)

print("Gradiente numérico:", gradiente_num)

print("Gradientes equivalentes:",
      np.allclose(gradiente, gradiente_num))


# =====================================================
# TESTE 6: CINCO SEMENTES ALEATÓRIAS
# =====================================================

print("\n========== TESTE 6: CINCO SEMENTES ==========")

def verificar_gradiente(seed):
    rng = np.random.default_rng(seed)

    X_teste = rng.normal(size=(20, 4))
    y_teste = rng.normal(size=20)
    w_teste = rng.normal(size=4)

    analitico = grad_mse(X_teste, y_teste, w_teste)

    numerico = grad_numerico(
        lambda pesos: mse(X_teste, y_teste, pesos),
        w_teste
    )

    return np.allclose(analitico, numerico)

resultados = (
    verificar_gradiente(0),
    verificar_gradiente(1),
    verificar_gradiente(2),
    verificar_gradiente(3),
    verificar_gradiente(4)
)

print("Resultados:", resultados)
print("Todos os testes passaram:", all(resultados))


# =====================================================
# TESTE 7: DESEMPENHO (10000 x 50)
# =====================================================

print("\n========== TESTE 7: DESEMPENHO ==========")

rng = np.random.default_rng(42)

X_grande = rng.normal(size=(10000, 50))
y_grande = rng.normal(size=10000)
w_grande = rng.normal(size=50)

inicio = time.perf_counter()

resultado = grad_mse(X_grande, y_grande, w_grande)

tempo = time.perf_counter() - inicio

print("Dimensão de X:", X_grande.shape)
print(f"Tempo de execução: {tempo:.6f} segundos")
print("Executou em menos de 1 segundo:", tempo < 1.0)

print("\n========== FIM DOS TESTES ==========")
