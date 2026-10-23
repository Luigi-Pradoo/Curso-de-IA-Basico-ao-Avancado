# Aula 6.9 — Árvore de Decisão do zero

> **Módulo 06 · ML na mão, com NumPy** · 2 h · Python e NumPy · Árvore própria × scikit-learn
> Material: [`Aula 6.9 - Árvore de Decisão.pdf`](Aula%206.9%20-%20%C3%81rvore%20de%20Decis%C3%A3o.pdf) (91 slides) + 🌳 **[laboratório interativo](arvore-interativa.html)**

Os três classificadores anteriores decidem com **pesos** (regressão logística), **distâncias** (KNN) ou **probabilidades** (Naive Bayes). A árvore de decisão decide com uma **sequência de perguntas** do tipo *"característica ≤ limiar?"* — exatamente como uma pessoa explicaria uma decisão. O ponto central da aula é que **nenhuma pergunta é escrita à mão**: o algoritmo testa todos os cortes possíveis, mede quão misturado cada lado fica (impureza de **Gini**), escolhe o melhor e repete o mesmo processo em cada lado — **recursividade**. Saber quando parar (critérios de parada e poda) é o que separa uma árvore que generaliza de uma que decora o treino.

```text
Árvore → Impureza → Gini → Avaliação dos cortes → Melhor divisão
       → Recursividade → Critérios de parada → Overfitting → Poda → Generalização
```
*Cada conceito resolve o problema deixado pelo anterior.*

---

## 🌳 Laboratório interativo — [`arvore-interativa.html`](arvore-interativa.html)

Abra o arquivo no navegador (duplo clique). Ele usa **os mesmos 24 clientes, a mesma divisão treino/teste e o mesmo algoritmo** do exercício guiado, e tudo reage junto:

| Você mexe em… | …e vê na hora |
|---|---|
| **Pontos do plano** (adicionar, arrastar, apagar clientes) | a árvore é retreinada; regiões de decisão, acurácias e curva de profundidade se atualizam |
| **`max_depth`, `min_samples_split`, `min_samples_leaf`** | a árvore cresce ou é podada; o motivo de cada folha ter parado aparece no nó |
| **Clique num nó** | o Gini calculado à mão (contagens → proporções → quadrados → 1 − soma), o Gini ponderado e o ganho do corte escolhido |
| **Busca do melhor corte** | o Gini ponderado de **todos** os limiares candidatos do nó, para Idade e Visitas; passe o mouse para pré-visualizar o corte no plano |
| **Mouse no plano** (ou Idade/Visitas digitados) | o percurso do cliente acende na árvore, pergunta por pergunta, até a folha |
| **Coluna do gráfico Profundidade × acurácia** | escolhe o `max_depth`: treino sempre sobe, teste para de melhorar |

> Também publicado online: <https://claude.ai/artifact/LpcemHYZ9kvEjmoeUM3esc> (precisa estar compartilhado para os alunos abrirem).

**Experimentos sugeridos para a aula:**
1. Com `max_depth = 3`, clique na raiz: confira que **Visitas ≤ 3,5** tem o menor Gini ponderado (0,3571) entre os 26 cortes testados.
2. Arraste o `max_depth` de 0 a 7 e observe o treino chegar a 100% enquanto o teste fica em 83,3% — **overfitting** visível.
3. Coloque `min_samples_leaf = 3` e veja quais cortes ficam riscados na tabela (isolam poucos pontos).
4. Adicione um cliente "estranho" (ex.: 60 anos, 1 visita, comprou) e veja a árvore criar uma região só para ele.

---

## Objetivos

1. Explicar raiz, nós, ramificações, folhas e profundidade.
2. Calcular a impureza de Gini à mão.
3. Calcular a impureza ponderada e o ganho.
4. Encontrar o melhor corte entre todos os candidatos.
5. Entender recursividade e caso base.
6. Construir a árvore com critérios de parada.
7. Fazer previsões e controlar overfitting com poda.
8. Implementar em NumPy e comparar com o scikit-learn.

---

## Teoria

### Anatomia de uma árvore
| Parte | O que é |
|---|---|
| **Raiz** | nó inicial (profundidade 0); recebe todas as amostras |
| **Nó interno** | faz uma pergunta: *característica ≤ limiar?* |
| **Ramificação** | caminho Sim (esquerda) ou Não (direita) |
| **Folha** | nó final: devolve a classe majoritária das amostras que chegaram nele |
| **Profundidade** | número de perguntas da raiz até o nó |

Cada amostra percorre **um único caminho**: perguntas de outros ramos nunca são feitas para ela.

### Impureza de Gini
$$\text{Gini} = 1 - \sum_i p_i^2$$

| Grupo | Proporções | Gini |
|---|---|---|
| A — 10 × classe 0 | p₀ = 1, p₁ = 0 | **0** (puro) |
| B — 5 / 5 | p₀ = p₁ = 0,5 | **0,5** (mistura máxima no caso binário) |
| C — 8 / 2 | p₀ = 0,8, p₁ = 0,2 | **0,32** |

### Gini ponderado e ganho
$$\text{Gini}_{\text{divisão}} = \frac{n_{esq}}{n}\,\text{Gini}_{esq} + \frac{n_{dir}}{n}\,\text{Gini}_{dir} \qquad \text{Ganho} = \text{Gini}_{pai} - \text{Gini}_{\text{divisão}}$$

A ponderação importa: num nó 50/50 com 100 amostras, separar **1 amostra** pura de 99 misturadas dá média simples ≈ 0,25 (parece ótimo), mas ponderado ≈ 0,495 (não ajudou quase nada).

### Melhor corte — exemplo dos slides
Seis clientes, só Idade: `[18, 22, 25, 40, 45, 50]`, `y = [0, 0, 1, 1, 1, 0]` (Gini da raiz 0,5). Os candidatos são os **pontos médios** entre valores consecutivos:

| Limiar | Esquerda | Direita | Gini esq | Gini dir | Ponderado | Ganho |
|---|---|---|---|---|---|---|
| 20 | [0] | [0, 1, 1, 1, 0] | 0 | 0,48 | 0,4 | 0,1 |
| **23,5** | **[0, 0]** | **[1, 1, 1, 0]** | **0** | **0,375** | **0,25** | **0,25** |
| 32,5 | [0, 0, 1] | [1, 1, 0] | 0,4444 | 0,4444 | 0,4444 | 0,0556 |
| 42,5 | [0, 0, 1, 1] | [1, 0] | 0,5 | 0,5 | 0,5 | 0 |
| 47,5 | [0, 0, 1, 1, 1] | [0] | 0,48 | 0 | 0,4 | 0,1 |

Vence **Idade ≤ 23,5** (menor Gini ponderado = maior ganho). Com mais características, repete-se a tabela para cada coluna de `X` e escolhe-se o menor valor geral.

### Recursividade
Cada lado do corte é **o mesmo problema, só que menor**. Uma função recursiva precisa de:
- **caso base** — situação simples em que ela para (aqui: virar folha);
- **chamada recursiva** — a função chama a si mesma com uma parte menor do problema (`depth + 1`).

A árvore dos seis clientes se constrói em 10 passos: Gini da raiz → avaliar cortes → raiz `Idade ≤ 23,5` → filhos → mesma busca no lado direito (`Idade ≤ 47,5`) → folhas puras encerram a recursão → classificar um cliente novo de 30 anos (→ classe 1).

### Critérios de parada (pré-poda)

| Critério | Teste no código | Efeito |
|---|---|---|
| Nó puro | `gini(y) == 0` | nada a separar |
| Profundidade máxima | `depth >= max_depth` | limita a altura da árvore |
| Mínimo para dividir | `len(y) < min_samples_split` | nós pequenos viram folha |
| Mínimo por folha | `len(y_esq) < min_samples_leaf` | descarta cortes que isolam poucos pontos |
| Sem ganho válido | `feature is None or ganho <= 0` | a pergunta não ajudaria |

A folha devolve `classes[np.argmax(contagens)]`.

### Overfitting e poda
Mais profundidade **sempre reduz a impureza de treino**, mas não garante generalização: as regiões de decisão viram retângulos cada vez menores, ajustados a pontos isolados.

| | Quando | Como | Parâmetros | Risco |
|---|---|---|---|---|
| **Pré-poda** | durante o treino | impede divisões | `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` | parar cedo demais |
| **Pós-poda** | depois da árvore completa | troca ramos por folhas | `ccp_alpha`: $R_\alpha(T) = R(T) + \alpha\lvert T\rvert$ | custo de crescer a árvore inteira |

**Poda por erro reduzido:** tenta trocar cada ramo por uma folha e aceita a troca se a **validação não cair**. Como a validação decidiu a poda, o número dela fica otimista — confirme no teste, uma única vez.

### Comparação com o `DecisionTreeClassifier`
Estruturas diferentes **não** significam implementação errada. Diferenças possíveis: empates entre cortes (a ordem de busca decide quem vence), limiares em `float32` no sklearn (ex.: `2.950000047683716`) e divisões com ganho zero que o sklearn ainda pode fazer.

---

## Exercício guiado — 18 etapas (`exercicio_arvore.ipynb`)

Dataset: **24 clientes**, duas características — **Idade** (anos) e **Visitas** à loja no mês — e o rótulo comprou (1) / não comprou (0), 12 de cada.

| Bloco | Etapas | Resultado de referência |
|---|---|---|
| **Preparação** | 01 imports · 02 dataset · 03 gráfico · 04 treino/teste (`test_size=0.25, random_state=42, stratify=y`) | treino `(18, 2)` [9 9] · teste `(6, 2)` [3 3] |
| **Matemática** | 05 Gini à mão · 06 `gini()` · 07 testar `Idade ≤ 30` · 08 `gini_ponderado()` | Gini de `[0,0,1,1,1]` = **0,48** |
| **Melhor corte** | 09 `melhor_divisao()` | **26 cortes** testados; melhor **Visitas ≤ 3,5**, ponderado **0,3571** (os 4 clientes com até 3 visitas nunca compraram) |
| **Estrutura** | 10 classe `No` · 11 recursividade · 12 classe `ArvoreDecisaoDoZero` (`fit`, `predict`, `score`, `_construir`, `_criar_folha`) | — |
| **Uso** | 13 treinar (`max_depth=3`) · 14 prever o teste e seguir um cliente · 15 acurácia e matriz de confusão · 16 imprimir a árvore · 17 regiões de decisão | treino **94,4%** · teste **83,3%** (5 de 6) · 4 folhas |
| **Conferência** | 18 comparar com `DecisionTreeClassifier` | mesmas previsões e mesma acurácia |

Árvore final (`max_depth = 3`):

```text
[Raiz] Visitas <= 3.5?
    [Sim] Classe 0  (4 amostras)
    [Não] Idade <= 53?
        [Sim] Idade <= 23.5?
            [Sim] Classe 0  (1 amostra)
            [Não] Classe 1  (10 amostras)
        [Não] Classe 0  (3 amostras)
```

Percurso do 2º cliente de teste (52 anos, 9 visitas): `Visitas 9 ≤ 3,5? Não` → `Idade 52 ≤ 53? Sim` → `Idade 52 ≤ 23,5? Não` → **classe 1**.
O único erro do teste é um cliente de 31 anos com 3 visitas que comprou: ele cai na folha *Visitas ≤ 3,5 → classe 0*.

---

## Erros comuns

| Erro | Causa | Correção |
|---|---|---|
| Confundir característica com classe | rótulo dentro de `X` | `X` só com entradas; a folha devolve um valor de `y` |
| Gini incorreto | contagens no lugar de proporções; sem o quadrado | `1 - np.sum((cont / len(y)) ** 2)` |
| Esquecer de ponderar | média simples dos dois filhos | pesos `n_esq/n` e `n_dir/n` |
| Escolher o **maior** Gini | objetivo invertido | menor Gini ponderado = maior ganho |
| Criar grupos vazios | limiar fora dos dados; sem checagem | pontos médios e `min_samples_leaf` |
| Índices incorretos | máscaras diferentes em `X` e `y`; `X[coluna]` | mesma máscara; sempre `X[:, coluna]` |
| Recursão sem caso base | esquecer os critérios de parada | devolver a folha antes de chamar de novo → senão `RecursionError` |
| Não respeitar a profundidade máxima | sem `depth + 1`, ou teste com `>` | `depth >= max_depth` |
| Classe errada na folha | `np.argmax(classes)` | `classes[np.argmax(contagens)]` |
| Teste na seleção de parâmetros | estimativa otimista | validação ou validação cruzada; teste só no fim |

## Desafio de casa — implemente sua própria árvore (`desafio_arvore_alunos.ipynb`)

**Requisitos:** receber `X` e `y` · impureza de Gini · buscar o melhor corte · criar nós internos · criar folhas · construção recursiva · critérios de parada · fazer previsões · calcular acurácia · imprimir a árvore · comparar com o scikit-learn.

**Entregas:** código completo e dataset utilizado · previsões e acurácia · estrutura da árvore impressa · gráfico das regiões de decisão · comparação com o scikit-learn · pequena análise dos resultados.

### Checklist de aprendizagem
- [ ] Explico raiz, nós, ramos e folhas
- [ ] Calculo o Gini à mão
- [ ] Calculo a impureza ponderada e o ganho
- [ ] Encontro o melhor corte
- [ ] Entendo recursividade e sei o que é um caso base
- [ ] Sei os critérios de parada
- [ ] Explico overfitting e poda
- [ ] Construo a árvore com NumPy
- [ ] Faço previsões à mão e no código
- [ ] Imprimo a árvore no terminal

## Leitura
- scikit-learn — [Guia de Árvores de Decisão](https://scikit-learn.org/stable/modules/tree.html)
- [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- Visualização: [`plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html) e [`export_text`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html)
- [Poda por complexidade de custo](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
- [Documentação do NumPy](https://numpy.org/doc/stable/)

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 6.9 - Árvore de Decisão.pdf` | slides da aula |
| `arvore-interativa.html` | laboratório interativo (abre direto no navegador, sem instalar nada) |

> Os notebooks `exercicio_arvore.ipynb` e `desafio_arvore_alunos.ipynb` citados nos slides ainda não estão nesta pasta.

---

**← Anterior:** [Aula 6.8 — Naive Bayes do zero](../Aula%206.8/README.md) · **Voltar ao módulo:** [Aula 6](../README.md)
