import matplotlib.pyplot as plt

#Grafico de linha

faltas = [1, 2, 3, 4, 5]
dias = [25, 27, 26, 30, 32]

plt.plot(faltas, dias)

plt.xlabel("Faltas")
plt.ylabel("Dias")
plt.title("Dias de aula por número de faltas")

plt.show()