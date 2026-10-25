# %% [markdown]
# # Etapa 1: Exploração dos Dados (EDA) — Acidentes PRF
# 
# Este script/notebook realiza a auditoria inicial dos dados brutos da Polícia Rodoviária Federal (`datatran`),
# identifica valores ausentes, anomalias e possíveis fontes de vazamento de dados (*data leakage*),
# e separa imediatamente o conjunto de teste de forma estratificada.

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configuração visual
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# %% [markdown]
# ## 1. Carregamento da Base de Dados Bruta
# Os arquivos CSV da PRF utilizam codificação `latin-1`, separador `;` e vírgula como separador decimal.

# %%
DATA_PATH = Path("data/raw/datatran2025.csv")

if not DATA_PATH.exists():
    # Caso o grupo opte pelo ano consolidado de 2024
    DATA_PATH = Path("data/raw/datatran2024.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Arquivo CSV não encontrado em 'data/raw/'. "
        "Baixe o CSV 'Agrupados por ocorrência' em: "
        "https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf "
        "e salve como 'data/raw/datatran2025.csv' ou 'data/raw/datatran2024.csv'."
    )

df_raw = pd.read_csv(
    DATA_PATH,
    sep=";",
    encoding="latin-1",
    decimal=",",
    low_memory=False
)

print(f"Dimensões da base bruta: {df_raw.shape[0]} linhas e {df_raw.shape[1]} colunas.")
df_raw.head()

# %% [markdown]
# ## 2. Auditoria de Colunas e Bloqueio de Data Leakage
# Removemos variáveis pós-evento (contagens de vítimas) que entregariam a resposta da variável-alvo.

# %%
TARGET_COL = "classificacao_acidente"

COLUNAS_VAZAMENTO = [
    "mortos",
    "feridos_graves",
    "feridos_leves",
    "ilesos",
    "feridos",
    "ignorados",
    "pessoas",
]

COLUNAS_ID_METADADOS = [
    "id",
    "data_inversa",
    "regional",
    "delegacia",
    "uop",
]

# Remove linhas onde a própria variável-alvo está ausente
df = df_raw.dropna(subset=[TARGET_COL]).copy()

# Padroniza strings de espaços em branco na variável-alvo
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

# Remove colunas de vazamento e metadados administrativos
colunas_para_remover = [c for c in (COLUNAS_VAZAMENTO + COLUNAS_ID_METADADOS) if c in df.columns]
df = df.drop(columns=colunas_para_remover)

print(f"Dimensões após remoção de leakage e IDs: {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("Colunas remanescentes:", list(df.columns))

# %% [markdown]
# ## 3. Separação Imediata do Conjunto de Teste (Avaliação Honesta)
# Separamos 20% dos dados para teste antes de qualquer decisão de imputação ou engenharia de atributos.

# %%
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[TARGET_COL]
)

print(f"Conjunto de Treino: {train_df.shape[0]} linhas")
print(f"Conjunto de Teste (intocado): {test_df.shape[0]} linhas")

# %% [markdown]
# ## 4. O que tem, o que falta e o que é estranho (Apenas no Conjunto de Treino)

# %%
print("\n--- 4.1. Distribuição da Variável-Alvo (Treino) ---")
dist_alvo = train_df[TARGET_COL].value_counts(dropna=False).to_frame(name="contagem")
dist_alvo["percentual (%)"] = (train_df[TARGET_COL].value_counts(normalize=True) * 100).round(2)
print(dist_alvo)

# %%
print("\n--- 4.2. Valores Ausentes Explícitos (NaN) e Implícitos ('Ignorado' / '(null)') ---")
relatorio_ausentes = pd.DataFrame({
    "tipo": train_df.dtypes,
    "nulos_NaN": train_df.isna().sum(),
    "pct_NaN (%)": (train_df.isna().mean() * 100).round(2),
    "strings_Ignorado_ou_null": train_df.apply(
        lambda col: col.astype(str).str.strip().isin(["Ignorado", "(null)", "Não Informado"]).sum()
    ),
    "cardinalidade_unicos": train_df.nunique(dropna=True)
})
print(relatorio_ausentes.sort_values(by="strings_Ignorado_ou_null", ascending=False))

# %%
print("\n--- 4.3. O que é estranho: Inspecionando 'tracado_via' (Multivalorado) e Numéricas ---")
if "tracado_via" in train_df.columns:
    print("Total de combinações únicas em 'tracado_via':", train_df["tracado_via"].nunique())
    print("Top 10 combinações mais frequentes em 'tracado_via':")
    print(train_df["tracado_via"].value_counts().head(10))

print("\nEstatísticas descritivas das variáveis numéricas (checando outliers e coordenadas):")
print(train_df.describe().T)