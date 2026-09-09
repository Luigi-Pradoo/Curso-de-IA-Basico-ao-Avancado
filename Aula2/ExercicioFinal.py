nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
notas = [8, 6, 9, 5, 7, 10]

aprovados = 0
reprovados = 0

# FOR
for nome in nomes:
    print("Aluno:", nome)

# WHILE
i = 0

while i < 6:

    if notas[i] >= 7:
        print(nomes[i], "- Aprovado")
        aprovados += 1
    else:
        print(nomes[i], "- Reprovado")
        reprovados += 1

    i += 1

# DICIONÁRIOS
alunos = [
    {"nome": "Evelyn", "nota": 8},
    {"nome": "Pedro", "nota": 6},
    {"nome": "Gustavo", "nota": 9},
    {"nome": "Mateus", "nota": 5},
    {"nome": "Felipe", "nota": 7},
    {"nome": "Marcelo", "nota": 10}
]

print("\nInformações dos alunos:")

for aluno in alunos:
    print(aluno)

# RESULTADO
print("\nResultado final:")
print("Total de alunos:", 6)
print("Aprovados:", aprovados)
print("Reprovados:", reprovados)