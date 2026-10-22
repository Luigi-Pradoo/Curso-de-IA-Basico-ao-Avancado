# Aula 1 — Introdução à IA, ambiente virtual e Git

> **Curso preparatório · Aula 1** · Fundamentos conceituais e ferramentas de trabalho
> Material: [`CURSO DE IA.pdf`](CURSO%20DE%20IA.pdf) (27 slides)

Primeira aula do curso. Ainda não tem código de modelo: o objetivo é deixar claro **o que é IA, onde Machine Learning e Deep Learning se encaixam** e preparar o ambiente que vamos usar no resto do curso (Python isolado num ambiente virtual e projeto versionado com Git/GitHub).

---

## Objetivos

Ao final da aula você consegue:

1. Explicar o que é Inteligência Artificial e como ela evoluiu até a IA generativa.
2. Diferenciar **IA ⊃ Machine Learning ⊃ Deep Learning**.
3. Citar os cinco principais tipos de aprendizado de máquina.
4. Dizer quando ML clássico é suficiente e quando Deep Learning compensa.
5. Criar um ambiente virtual e explicar por que ele é obrigatório em projetos de IA.
6. Criar um repositório no GitHub e usar os comandos básicos do Git.

---

## Conteúdo

### 1. O que é IA
Área da computação que busca criar sistemas capazes de realizar tarefas que normalmente exigiriam inteligência humana: reconhecer objetos, compreender linguagem, aprender, tomar decisões e resolver problemas.

**Linha do tempo**

| Época | Marco | Ideia central |
|---|---|---|
| Anos 50 | Inteligência Artificial | inteligência humana reproduzida por máquinas |
| Anos 80 | Machine Learning | sistemas que aprendem a partir de dados históricos |
| 2010 | Deep Learning | modelos baseados em redes neurais profundas |
| 2020 | IA Generativa | modelos capazes de **criar** novos conteúdos |

### 2. Machine Learning
Em vez de programar todas as regras à mão, o modelo **aprende as relações entre os dados e os resultados**. Depois do treino, ele faz previsões sobre dados que nunca viu.

### 3. Tipos de aprendizado

| Tipo | Como aprende | Exemplo |
|---|---|---|
| **Supervisionado** | exemplos com a resposta certa (rótulo) | foto → "cachorro", "gato", "pássaro" |
| **Não supervisionado** | encontra grupos/padrões sem rótulo | agrupar clientes parecidos |
| **Semi-supervisionado** | poucos dados rotulados + muitos sem rótulo | rotular uma parte das imagens e aproveitar o resto |
| **Auto-supervisionado** | o próprio dado gera a tarefa (esconde uma parte e tenta prever) | imagem mascarada → prever o pedaço que falta |
| **Por reforço** | agente age num ambiente e recebe recompensa ou penalidade | robô/jogo aprendendo por tentativa e erro |

### 4. Deep Learning
Subárea do ML baseada em **redes neurais com muitas camadas**. Aprende as características (features) direto dos dados, construindo representações hierárquicas — de padrões simples a complexos. É a base da Visão Computacional, do PLN e da IA Generativa modernos, e brilha quando há muitos dados e poder computacional.

### 5. ML × DL

| Machine Learning | Deep Learning |
|---|---|
| Vários tipos de algoritmo | Principalmente redes neurais profundas |
| Depende muito de **engenharia de features** | Aprende as features sozinho |
| Pode funcionar bem com menos dados | Brilha com grandes volumes de dados |
| Menos poder computacional | Normalmente exige GPU e mais tempo |
| Ótimo para **dados tabulares** | Forte em imagem, áudio, texto e vídeo |

### 6. Aplicações
👁️ Visão Computacional · 💬 PLN · 🤖 Robótica · 🏥 Saúde · 🚗 Veículos autônomos · 🎨 IA Generativa

### 7. Ambiente virtual
Um "caixinha separada" para cada projeto: instala só as dependências necessárias, controla as versões e evita que um projeto quebre o outro. Em projetos reais de IA é boa prática essencial.

```bash
# criar e ativar (Windows)
python -m venv .venv
.venv\Scripts\activate

# criar e ativar (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# instalar as bibliotecas do curso
pip install numpy pandas matplotlib scikit-learn scipy
```

> 💡 Neste repositório o script [`RECRIAR_AMBIENTE.bat`](../RECRIAR_AMBIENTE.bat) recria o ambiente do curso fora do OneDrive (em `C:\dev\venvs\curso-ia`) e aponta o VS Code para ele.

### 8. Git e GitHub
- **Git** = controla as versões (histórico inteligente do projeto).
- **GitHub** = hospeda e compartilha os projetos na nuvem (e vira portfólio).

| Comando | Para que serve |
|---|---|
| `git clone URL` | baixar um repositório do GitHub |
| `git status` | ver o que mudou |
| `git add .` | preparar as alterações |
| `git commit -m "mensagem"` | salvar a alteração no histórico |
| `git push` | enviar as mudanças para o GitHub |
| `git pull` | atualizar seu projeto com o que está no GitHub |
| `git branch` | criar/listar ramificações |
| `git checkout` | trocar de branch |
| `git merge` | unir as mudanças de uma branch |
| *Fork* (no GitHub) | criar sua cópia do projeto para contribuir |
| *Pull Request* (no GitHub) | pedir que suas alterações sejam integradas |

Primeiro push de um projeto novo:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<usuario>/<repo>.git
git push -u origin main
```

---

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `CURSO DE IA.pdf` | slides da aula |

---

**Próxima aula →** [Aula 2 — Python para IA](../Aula2/README.md)
