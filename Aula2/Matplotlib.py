import matplotlib.pyplot as plt

nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
faltas = [2, 5, 1, 3, 7, 4]

# Gráfico de barras
plt.bar(nomes, faltas)

plt.xlabel("Alunos")
plt.ylabel("Dias de falta")
plt.title("Faltas dos alunos")

plt.show()


#Grafico de linhas

nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
faltas = [2, 5, 1, 3, 7, 4]

plt.plot(nomes, faltas)

plt.xlabel("Alunos")
plt.ylabel("Dias de falta")
plt.title("Faltas dos alunos")

plt.show()

#Histograma

faltas = [2, 5, 1, 3, 7, 4]

plt.hist(faltas)

plt.xlabel("Dias de falta")
plt.ylabel("Quantidade de alunos")
plt.title("Distribuição das faltas")

plt.show()

#dispersao 

aulas = [20, 21, 19, 22, 23, 24]
faltas = [2, 5, 1, 3, 7, 4]

plt.scatter(aulas, faltas)

plt.xlabel("Aulas")
plt.ylabel("Dias de falta")
plt.title("Aulas x Dias de falta")

plt.show()


