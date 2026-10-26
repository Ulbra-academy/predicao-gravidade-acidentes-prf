# Diário de Bordo - Projeto Prático de Machine Learning (G2)

**Tema:** Predição de Gravidade de Acidentes em Rodovias Federais Brasileiras (PRF)  
**Fonte dos Dados:** [Portal de Dados Abertos da PRF (Gov.br)](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) - Base `datatran` (Agrupados por ocorrência)  
**Link para Download do CSV:** [Dados Abertos PRF - Seção BAT (Acidentes Agrupados por Ocorrência)](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)  
**Dicionário de Dados Oficial:** [Dicionário de Dados - Acidentes PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dicionario-acidentes)

## Integrantes
* Danilo Fernandes Araujo Gama
* [Nome do Integrante 2]
* Lucas da Silva Nascimento

## Cronograma de Entregas

| Marco de Entrega | Data Limite |
| :--- | :--- |
| Escolha do dataset e definição do problema | 26/10/2026 |
| Entrega final do repositório | 07/12/2026 |
| Apresentação final | 12/12/2026 |

---

## Sessão 01 - 25/10/2026: Triagem de Datasets, Descarte de Alternativas e Definição do Escopo

### 1. O que tentamos
Reunimos o grupo para definir a pergunta preditiva e auditar conjuntos de dados no Kaggle, no UCI Machine Learning Repository e no Portal Brasileiro de Dados Abertos frente aos critérios da disciplina (mínimo de $1.000$ linhas e $6$ atributos, presença de desafios reais de pré-processamento e ineditismo em relação aos conjuntos usados em aula).

Avaliamos inicialmente quatro candidatos:
1. **Wine Quality e Student Performance (UCI):** Analisados preliminarmente para regressão de qualidade e notas escolares.
2. **Heart Disease / Credit Card Default (UCI):** Avaliados para classificação binária.
3. **Consumo de Energia Industrial - Steel Industry (UCI):** Avaliado para regressão de consumo energético (`Usage_kWh`).
4. **Acidentes em Rodovias Federais - PRF (`datatran` / Gov.br):** Inspecionado quanto à volumetria anual (cerca de $65.000$ ocorrências por ano e $30$ colunas), tipagem de variáveis, presença de valores ausentes e desafios reais de engenharia de atributos.

### 2. O que deu errado / O que descartamos (e por quê)
* **Tentativa 1 - Wine Quality e Student Performance (UCI):** Descartados. Apesar de bem documentados, os conjuntos são excessivamente limpos (praticamente sem dados ausentes reais) e muito recorrentes em listas de exercícios, o que empobreceria nossa **Etapa 2 (Preparação justificada)** e aumentaria o risco de colisão de tema com outros grupos.
* **Tentativa 2 - Base da PRF agrupada por pessoa vs. agrupada por ocorrência:** Ao explorar o [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), vimos que existem três formatos de CSV por ano: *Agrupados por ocorrência*, *Agrupados por pessoa* e *Agrupados por pessoa - Todas as causas e tipos*. Percebemos que usar a tabela agrupada por pessoa geraria duplicação de contexto da via e do clima para um mesmo acidente (`id`), o que violaria a independência entre observações caso o mesmo `id` de acidente caísse simultaneamente nos conjuntos de treino e de teste. Por isso, descartamos a visão por pessoa e fixamos a base **Agrupados por ocorrência** (`datatran`).
* **Tentativa 3 - Risco de Vazamento de Dados (*Data Leakage*) na base bruta da PRF:** Ao inspecionar o [Dicionário de Dados da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dicionario-acidentes), identificamos que a planilha bruta traz contagens registradas após o desfecho do evento (`mortos`, `feridos_graves`, `feridos_leves`, `ilesos`, `feridos`, `ignorados`). Se mantivéssemos essas colunas como preditoras para prever `classificacao_acidente` (*Sem Vítimas*, *Com Vítimas Feridas*, *Com Vítimas Fatais*), os modelos atingiriam $100\%$ de acurácia artificialmente por *data leakage*, já que a classificação deriva diretamente dessas contagens.
* **Tentativa 4 - Leitura padrão do CSV com `pd.read_csv()`:** Na primeira tentativa rápida de carregar o CSV da PRF com os parâmetros padrão do `pandas`, ocorreu erro de decodificação (`UnicodeDecodeError` em `utf-8`) e falha de separação de colunas, pois o arquivo governamental utiliza codificação `latin-1` (`ISO-8859-1`), delimitador ponto e vírgula (`;`) e vírgula (`,`) como separador decimal nas colunas `km`, `latitude` e `longitude`.

### 3. O que decidimos e por quê
1. **Escolha oficial do Dataset:** Base oficial de **Acidentes em Rodovias Federais da PRF (Agrupados por ocorrência)**, utilizando o recorte anual consolidado (`datatran2025.csv` / `datatran2024.csv`), que supera amplamente o requisito mínimo (mais de $60.000$ linhas e mais de $15$ atributos preditivos válidos).
2. **Pergunta Preditiva (Classificação):**  
   *"É possível prever a gravidade de um acidente rodoviário federal a partir das condições meteorológicas, fase do dia, características geométricas da via e tipo de acidente?"*
   * **Variável-alvo (`target`):** `classificacao_acidente` (será avaliada na Etapa 1 tanto em sua forma multiclasse original quanto binarizada para medir o impacto do desbalanceamento da classe *Com Vítimas Fatais*).
3. **Bloqueio de *Data Leakage* e Seleção Preliminar de Atributos:**
   * **Removidas imediatamente:** `id`, `mortos`, `feridos_graves`, `feridos_leves`, `ilesos`, `feridos`, `ignorados`, `pessoas` (pós-evento) e metadados administrativos (`regional`, `delegacia`, `uop`).
   * **Mantidas para exploração e preparação:** `dia_semana`, `horario`, `uf`, `br`, `km`, `causa_acidente`, `tipo_acidente`, `fase_dia`, `sentido_via`, `condicao_metereologica`, `tipo_pista`, `tracado_via`, `uso_solo`, `veiculos`, `latitude`, `longitude`.
4. **Planejamento dos 4 Modelos Obrigatórios:**
   * **Modelo Linear:** `LogisticRegression` (exigirá `StandardScaler` nas numéricas e `OneHotEncoder`/`TargetEncoder` nas categóricas).
   * **Árvore de Decisão:** `DecisionTreeClassifier` (para avaliar *overfitting* sem poda vs. com restrição de `max_depth` e `min_samples_leaf`).
   * **Método de Conjunto (Ensemble):** `RandomForestClassifier` ou `HistGradientBoostingClassifier` / `LightGBM` (Alvo #1 da busca de hiperparâmetros).
   * **Escolha Livre:** `MLPClassifier` (Rede Neural Multilayer Perceptron) (Alvo #2 da busca de hiperparâmetros).

### 4. Próximos passos (Sessão 02)
* Comunicar oficialmente o tema e o dataset ao professor até **26/10/2026**.
* Executar o script `notebooks/01_exploracao.py` sobre o CSV baixado para quantificar valores ausentes, inspecionar a multivaloração da coluna `tracado_via` (que concatena múltiplos atributos separados por `;`) e realizar a separação estratificada entre treino ($80\%$) e teste ($20\%$).