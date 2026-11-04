import numpy as np

#Versão 1: laço explícito

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
m = len(x)                          # m = 3

b0, b1 = 0.0, 0.0                   # β₀ e β₁ iniciais
alpha = 0.1                         # α

for t in range(1000):               # t: iteração
    y_hat = b0 + b1 * x             # ŷ = β₀ + β₁x
    erro = y_hat - y                # ŷ - y
    grad_b0 = erro.sum() / m            # ∂J/∂β₀
    grad_b1 = (erro * x).sum() / m      # ∂J/∂β₁
    b0 = b0 - alpha * grad_b0
    b1 = b1 - alpha * grad_b1

print(f'b0 = {b0:.4f}   b1 = {b1:.4f}')

#Versão 2: vetorizada (com montagem de X e registro da trajetória)

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
m = len(x)

# Montagem de X: coluna de 1s (para β₀) + coluna de x (para β₁) → (m, n+1)
X = np.column_stack([np.ones(m), x])
beta = np.zeros(X.shape[1])          # [β₀, β₁] = [0, 0]

alpha, epocas, tol = 0.1, 1000, 1e-8
historico = []
trajetoria = [beta.copy()]                       # antes do laço

for t in range(epocas):
    y_hat = X @ beta                             # ŷ = Xβ  (m,)
    erro = y_hat - y                             # ŷ - y   (m,)
    custo = (erro @ erro) / (2 * m)              # J(β)    escalar
    historico.append(custo)                      # historico[0] = custo inicial
    gradiente = (X.T @ erro) / m                 # ∇J = Xᵀ(ŷ - y)/m  (n+1,)
    beta = beta - alpha * gradiente              # β ← β - α∇J
    trajetoria.append(beta.copy())               # no laço, após atualizar
    if t > 0 and abs(historico[-2] - historico[-1]) < tol:
        break

traj = np.array(trajetoria)                      # (épocas + 1, 2)

print(f'Épocas: {len(historico)}   beta = {beta}')
# Conferência: 1ª passagem → custo 9.3333 e beta = [0.4, 0.9333]


#Calculando J em uma grade para desenhar o mapa
import matplotlib.pyplot as plt

B0, B1 = np.meshgrid(np.linspace(-1, 1.5, 200),
                     np.linspace(-0.5, 3, 200))      # (200, 200)
E = B0[..., None] + B1[..., None] * x - y           # (200, 200, m)
J_grade = (E ** 2).sum(axis=-1) / (2 * m)           # (200, 200)

plt.contour(B0, B1, J_grade, levels=20)
plt.plot(traj[:, 0], traj[:, 1], 'o-')
plt.xlabel('β₀'); plt.ylabel('β₁')
plt.show()

# Versão 3D da mesma grade:
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')
ax3d.plot_surface(B0, B1, J_grade, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('β₀'); ax3d.set_ylabel('β₁'); ax3d.set_zlabel('J')
plt.show()

# Animando a trajetória com FuncAnimation

from matplotlib.animation import FuncAnimation
from IPython.display import Image

fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=20)
caminho, = ax.plot([], [], 'o-', color='C1')
ax.set_xlabel('β₀')
ax.set_ylabel('β₁')

def atualizar(k):                                # k: quadro = iteração
    caminho.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
    J = ((X @ traj[k] - y) ** 2).sum() / (2 * m)
    ax.set_title(f't = {k}   J = {J:.4f}')
    return caminho,

anim = FuncAnimation(fig, atualizar, frames=range(0, len(traj), 5),
                     interval=100)
anim.save('descida.gif', writer='pillow', fps=10)   # salva o GIF na pasta do notebook

plt.close(fig)                   # evita mostrar o quadro estático extra
Image(filename='descida.gif')    # exibe o GIF salvo (funciona no VS Code, Jupyter e Colab)