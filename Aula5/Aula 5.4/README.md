# Aula 5.4 — Feature engineering: criação, codificação e escalonamento

> **Módulo 05 · Fundamentos práticos do ML** · 2 h · Dataset: **Titanic** (continuação da 5.3)
> Material: [`Aula 5.4 Feature Engineering.pdf`](Aula%205.4%20Feature%20Engineering.pdf) (51 slides) + [`Exercicio.py`](Exercicio.py)

Machine learning não é só escolher o algoritmo: **a representação dos dados influencia o resultado tanto quanto o modelo**. Nesta aula o modelo fica constante (regressão logística) justamente para que a única coisa que mude seja a representação. Partimos do absoluto básico — o que é uma feature — e chegamos a target encoding, transformações de distribuição e um transformer próprio dentro do Pipeline. A regra que atravessa tudo: **cada feature nasce de uma hipótese escrita antes do código e só fica se o ganho medido pagar.**

`Dado bruto → Feature → Transformação → Representação → Modelo → Métrica`

---

## Objetivos

1. Explicar observação, variável, feature e target, e classificar o tipo de cada coluna.
2. Criar uma feature a partir de uma hipótese escrita antes do código.
3. Escolher entre Min-Max, Standard e Robust olhando a distribuição da coluna.
4. Codificar categóricas com one-hot, ordinal e target encoding, sabendo o custo de cada um.
5. Identificar vazamento numa transformação e corrigir a ordem das etapas.
6. Medir o ganho de cada feature contra um baseline e descartar as que não pagam.
7. Montar um Pipeline com criação → imputação → codificação → escalonamento, e escrever um transformer próprio.

---

## Teoria

### Parte 1 — Vocabulário
- **Observação** = uma linha (um passageiro). **Variável** = uma coluna.
- **Feature (X)** = entrada disponível no momento da previsão. **Target (y)** = o que queremos prever.
- Nenhuma coluna é feature por natureza: o papel vem do problema.

| Tipo | Exemplo | Tratamento típico |
|---|---|---|
| Numérica contínua | idade, tarifa | imputação + escalonamento |
| Numérica discreta | nº de filhos a bordo | numérica ou faixas |
| Categórica nominal | porto de embarque | one-hot |
| Categórica ordinal | classe (1ª, 2ª, 3ª) | ordinal ou one-hot |

### Parte 2 — Por que criar features (e quando piora)
- A informação costuma estar **implícita** (o IMC não está em nenhuma coluna, mas resume peso e altura).
- Com 891 linhas o modelo não descobre interações sozinho; a feature entrega a interação de graça.
- Mas features **redundantes** desestabilizam coeficientes e **muitas colunas para poucas linhas** levam o modelo a decorar.

**As quatro features da aula e suas hipóteses**

| Feature | Como é criada | Hipótese |
|---|---|---|
| `FamilySize` | `SibSp + Parch + 1` | família média embarca junto no bote; família muito grande se perde |
| `IsAlone` | `FamilySize == 1` | viajar sozinho muda a prioridade no resgate |
| `FarePerPerson` | `Fare / FamilySize` | a tarifa é do bilhete, não da pessoa |
| `Title` | extraído de `Name` (Mr, Mrs, Miss, Master, Rare) | o tratamento social resume idade, sexo e status |

### Parte 3 — O ciclo de uma feature
`Hipótese → Feature → Treinamento → Validação → Métrica → Decisão`
Mesmo modelo, mesma semente, validação cruzada (nunca o teste), média **e desvio**, e o **limiar de ganho declarado antes de ver os números**.

### Parte 4 — Escalonamento
Idade (dezenas) e salário (milhares) na mesma distância: o salário domina. **Exigem escala:** KNN, K-Means, SVM-RBF, PCA, Ridge/Lasso, logística regularizada, redes neurais. **Indiferentes:** árvores, Random Forest, Gradient Boosting, Naive Bayes.

Mesma coluna com um outlier (10, 12, 13, 14, 15, **500**):
- **Min-Max** espreme os cinco valores normais perto de 0.
- **Standard** é puxado pela média e pelo desvio inflados.
- **Robust** (mediana e IQR) mantém os normais bem espalhados e isola o outlier.

### Parte 5 — Codificação

| Situação | Método |
|---|---|
| Ordem real na variável | `OrdinalEncoder(categories=[[...]])` — passe a ordem à mão |
| Sem ordem, até ~15 categorias | `OneHotEncoder` |
| Muitas categorias com relação clara com o alvo | `TargetEncoder` (com validação interna) |
| Frequência importa | count / frequency encoding |
| Cardinalidade explosiva (CEP, ID, URL) | hashing ou agrupar raras |

⚠️ **Target encoding vaza** se a média do alvo for calculada antes do split ou com a própria linha dentro. Use `TargetEncoder(smooth="auto", cv=5)` **dentro** do Pipeline.

### Avançado
- **Discretização** (`KBinsDiscretizer`) — faixa etária permite efeito não linear num modelo linear.
- **Transformações de distribuição** — `np.log1p`, `PowerTransformer` (Box-Cox / Yeo-Johnson), `QuantileTransformer`.
- **Transformer próprio** — `BaseEstimator` + `TransformerMixin`, com `fit` que devolve `self`.

### Parte 6 — Ordem no Pipeline
`1 Dados → 2 Feature engineering → 3 Imputação → 4 Encoding → 5 Scaling → 6 Modelo`

---

## Exercício guiado — 12 passos ([`Exercicio.py`](Exercicio.py))

| Passos | O que fazem |
|---|---|
| 1–4 | importar, separar `X`/`y`, criar (vira o transformer `Features`), dividir **antes de qualquer fit** |
| 5–8 | transformação (`Features`), encoding (one-hot com `min_frequency=10`), scaling (`RobustScaler`), treino (logística) |
| 9–12 | medir o baseline, adicionar **uma feature por vez**, comparar com o limiar (+0,002) e decidir |

**Resultado de referência** (regressão logística, `StratifiedKFold` de 5 dobras, semente 42):

| Conjunto | Acurácia CV | Ganho | Decisão |
|---|---|---|---|
| Baseline (7 colunas) | 0,792 | — | referência |
| + FamilySize | 0,801 | +0,009 | mantém |
| + IsAlone | 0,800 | −0,001 | descarta (redundante) |
| + FarePerPerson | 0,805 | +0,004 | mantém |
| + Title | 0,819 | +0,014 | mantém |

Três de quatro aprovadas, ~2,7 pontos de ganho acumulado; `Title` foi a mais forte. **Ressalva honesta:** com desvio entre dobras na casa de 0,02, um ganho de 0,004 é fraco — reportar o desvio junto com a média faz parte da resposta.

### Como rodar
```bash
cd "Aula5/Aula 5.4"
python Exercicio.py
```
Lê o Titanic de um CSV público no GitHub (precisa de internet).

---

## Erros comuns (dez armadilhas)
1. Escalar antes do split.
2. Usar o target para criar feature.
3. Target encoding sem dobra.
4. Ordinal em variável nominal (ordem alfabética vira "hierarquia").
5. Feature sem hipótese.
6. Escolher feature pelo teste.
7. Transformação diferente no teste.
8. Informação do futuro.
9. Ignorar outliers ao usar Min-Max.
10. Tomar ganho pequeno como prova (0,2 ponto com desvio de 2 pontos não é nada).

## Desafio de casa
Melhore o modelo do Titanic com **pelo menos três features novas**, cada uma precedida de uma hipótese em uma frase. Meça o ganho de cada uma contra o baseline (mesma semente, mesmas dobras), registre numa tabela e diga qual teve maior impacto e por quê.
**Critérios de aceite:** Pipeline executa de cima a baixo sem erro · sem vazamento evidente · três features justificadas por hipótese · baseline registrado · comparação em tabela · métrica adequada · teste aberto uma única vez, na última célula.

## Leitura
`StandardScaler` · `RobustScaler` · `OrdinalEncoder` · `OneHotEncoder` · `TargetEncoder` · `ColumnTransformer` · guia *Preprocessing data* — documentação do scikit-learn.

## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `Aula 5.4 Feature Engineering.pdf` | slides da aula |
| `Exercicio.py` | notebook completo em 12 passos |

---

**← Anterior:** [Aula 5.3 — Pipeline e ColumnTransformer](../Aula%205.3/README.md) · **Próxima →** [Aula 5.5 — Seleção de features](../Aula%205.5/README.md)
