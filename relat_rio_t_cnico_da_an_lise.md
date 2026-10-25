# Relatório Técnico — Predição de Gravidade de Acidentes em Rodovias Federais (PRF)

**Disciplina:** Aprendizado de Máquina (G2)  
**Integrantes:** Danilo Fernandes Araujo Gama, [Nome do Integrante 2], [Nome do Integrante 3]  
**Data de Entrega:** 07/12/2026

---

## 1. Definição do Problema e Descrição dos Dados

### 1.1. Contexto e Pergunta Preditiva
As rodovias federais brasileiras concentram grande parte do fluxo de cargas e passageiros do país. Neste trabalho, investigamos a seguinte pergunta preditiva de **classificação supervisionada**:
> *Dado o registro inicial das circunstâncias ambientais, temporais e viárias de um acidente em rodovia federal, é possível prever o nível de gravidade da ocorrência (`classificacao_acidente`)?*

### 1.2. Fonte e Características da Base Bruta
* **Fonte:** Polícia Rodoviária Federal (PRF) — [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf).
* **Unidade amostral:** Acidentes agrupados por ocorrência (`datatran`), onde cada linha representa um único evento viário.
* **Critérios de exclusão a priori (Prevenção de *Data Leakage*):** As variáveis `mortos`, `feridos_graves`, `feridos_leves`, `ilesos`, `feridos`, `ignorados` e `pessoas` determinam algebricamente a variável-alvo `classificacao_acidente` e só são consolidadas após o atendimento completo da ocorrência. Portanto, foram excluídas antes de qualquer etapa de modelagem.

---

## 2. Etapa 1: Exploração dos Dados (EDA)

*(Seção estruturada para consolidação dos resultados gerados em `notebooks/01_exploracao.py`)*

### 2.1. Volumetria e Distribuição da Variável-Alvo
* **Total de linhas e atributos brutos:** *[A preencher após execução do notebook 01]*
* **Proporção das classes em `classificacao_acidente`:**
  * *Com Vítimas Feridas:* *[X]%*
  * *Sem Vítimas:* *[Y]%*
  * *Com Vítimas Fatais:* *[Z]%*

### 2.2. Qualidade dos Dados: O que falta e o que é estranho
1. **Valores Ausentes e Categorias de Não-Preenchimento:** Identificação de valores `NaN` em `br`, `km`, `classificacao_acidente`, além de strings como `"Ignorado"` ou `"(null)"` em variáveis categóricas (`fase_dia`, `condicao_metereologica`, `sentido_via`).
2. **Anomalias em Coordenadas Geográficas e Quilometragem:** Presença de coordenadas `latitude` e `longitude` fora do território brasileiro ou zeradas, além de `km` com valores extremos.
3. **Alta Cardinalidade e Atributos Multivalorados:** A coluna `tracado_via` armazena combinações concatenadas por ponto e vírgula (ex.: `"Aclive;Curva;Em Obras"`), gerando centenas de categorias raras se tratada de forma ingênua.
4. **Codificação de Horário:** A variável `horario` é fornecida como string (`HH:MM:SS`) e possui natureza cíclica (ex.: $23\text{h}59$ está próximo de $00\text{h}05$).

---

## 3. Etapa 2: Preparação dos Dados

*(Seção estruturada para consolidação das justificativas técnicas implementadas em `notebooks/02_preparacao.py`)*

### 3.1. Isolamento Imediato do Conjunto de Teste
Antes de calcular qualquer estatística de imputação ou escalonamento, o conjunto de dados foi dividido de forma estratificada (`StratifiedShuffleSplit` / `train_test_split` com `stratify=y`) em:
* **Conjunto de Treino ($80\%$):** Usado para exploração profunda, ajuste de transformadores (`Pipeline`) e validação cruzada.
* **Conjunto de Teste ($20\%$):** Mantido intocado até a avaliação final (Etapa 4).

### 3.2. Tratamento de Dados Ausentes (Justificativa)
* **Numéricas (`km`, `latitude`, `longitude`):** Imputação pela **mediana** via `SimpleImputer(strategy="median")`, pois a distribuição espacial apresenta assimetria e *outliers* que distorceriam a média aritmética.
* **Categóricas (`condicao_metereologica`, `tipo_pista`, etc.):** Padronização de termos `"Ignorado"`/`NaN` em uma categoria explícita `"Desconhecido"` ou imputação pela moda, preservando a informação de que o agente não pôde aferir a condição no local.

### 3.3. Tratamento de Variáveis Categóricas e Engenharia de Atributos
* **Transformação Cíclica de Horário:** Conversão da hora $h \in [0, 23]$ em componentes trigonométricos para preservar a continuidade à meia-noite:
  $$\text{hora\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hora\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
* **Simplificação de `tracado_via`:** Extração de indicadores binários (*flags*) para os elementos críticos da via (`tem_curva`, `tem_aclive_declive`, `tem_cruzamento`, `em_obras`), evitando explosão dimensional.
* **Codificação (*Encoding*):** Uso de `OneHotEncoder(handle_unknown="ignore")` para variáveis de baixa cardinalidade (`fase_dia`, `tipo_pista`, `uso_solo`, `dia_semana`) e `TargetEncoder` (ou agrupamento de categorias infrequentes com `min_frequency`) para variáveis de alta cardinalidade (`uf`, `causa_acidente`, `tipo_acidente`).

### 3.4. Escalonamento de Variáveis Numéricas
* Aplicação de `StandardScaler` (padronização com média $\mu = 0$ e desvio-padrão $\sigma = 1$) sobre as variáveis contínuas e de contagem (`km`, `veiculos`, `latitude`, `longitude`).
* **Justificativa:** Indispensável para a convergência e penalização justa dos coeficientes na **Regressão Logística** (regularização $L_2$) e nos pesos da **Rede Neural (MLP)**, embora invariante para modelos baseados em árvores.

---

## 4. Etapas 3 e 5: Modelagem e Busca de Hiperparâmetros

Avaliamos quatro famílias distintas de algoritmos encapsuladas em `Pipeline` do `scikit-learn` para impedir vazamento entre *folds*:

1. **Modelo Linear:** `LogisticRegression(class_weight="balanced", max_iter=1000)`
2. **Árvore de Decisão:** `DecisionTreeClassifier(class_weight="balanced")`
3. **Método de Conjunto (Ensemble):** `HistGradientBoostingClassifier` / `RandomForestClassifier` *(Alvo #1 de Tuning)*
4. **Escolha Livre:** `MLPClassifier` (Rede Neural Multilayer Perceptron) *(Alvo #2 de Tuning)*

---

## 5. Etapa 4: Avaliação Honesta e Discussão dos Resultados

### 5.1. Estratégia de Validação e Métricas Escolhidas
* **Validação Cruzada:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` executado exclusivamente sobre o conjunto de treino.
* **Métricas Principais:** Como a distribuição de gravidade dos acidentes é desbalanceada, a **Acurácia** isolada é enganosa. Adotamos **F1-Macro**, **Recall da classe grave/fatal** e **ROC-AUC (One-vs-Rest / Binário)** como métricas primárias de decisão.

### 5.2. Tabela Comparativa (Validação Cruzada vs. Conjunto de Teste)

| Modelo | F1-Macro (CV Média $\pm$ DP) | ROC-AUC (CV Média $\pm$ DP) | F1-Macro (Teste Final) | ROC-AUC (Teste Final) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Regressão Logística (Linear) | *A preencher* | *A preencher* | *A preencher* | *A preencher* |
| 2. Árvore de Decisão | *A preencher* | *A preencher* | *A preencher* | *A preencher* |
| 3. Ensemble (Pós-Tuning) | *A preencher* | *A preencher* | *A preencher* | *A preencher* |
| 4. Escolha Livre — MLP (Pós-Tuning) | *A preencher* | *A preencher* | *A preencher* | *A preencher* |

### 5.3. Conclusões e Limitações
*(Análise crítica dos erros da matriz de confusão e importância das variáveis a ser preenchida após a Etapa 4).*