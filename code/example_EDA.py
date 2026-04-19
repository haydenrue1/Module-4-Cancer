#Statment of AI usage: Chapt gpt was used to troubleshoot errors in pca data matching and generate clinical data plots.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %%
# Load data
####################################################
data = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

metadata_df = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

print("Data shape:", data.shape)
print("Metadata shape:", metadata_df.shape)

# %%
# Robust TCGA matching
####################################################
data_short = data.columns.str[:12]
meta_short = metadata_df.index.str[:12]

data_map = dict(zip(data_short, data.columns))
meta_map = dict(zip(meta_short, metadata_df.index))

common_ids = set(data_short).intersection(set(meta_short))
print("Matched patients:", len(common_ids))

data_cols_final = [data_map[i] for i in common_ids]
meta_idx_final = [meta_map[i] for i in common_ids]

data = data[data_cols_final]
metadata_df = metadata_df.loc[meta_idx_final]

# %%
# Subset to SKCM
####################################################
cancer_type = 'SKCM'

cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
common_samples = data.columns.intersection(cancer_samples)

print("Matched SKCM samples:", len(common_samples))

SKCM_data = data[common_samples]
SKCM_metadata = metadata_df.loc[common_samples]

# %%
# Gene selection (ensure MMP9 included)
####################################################
gene_variance = SKCM_data.var(axis=1)
top_genes = gene_variance.sort_values(ascending=False).head(150).index

genes = list(set(top_genes).union({'MMP9'}))
SKCM_gene_data = SKCM_data.loc[genes]

print("Gene data shape:", SKCM_gene_data.shape)

# %%
# Merge expression + metadata
####################################################
SKCM_merged = SKCM_gene_data.T.merge(
    SKCM_metadata, left_index=True, right_index=True
)

# %%
# Clean missing values for clinical analysis
####################################################
SKCM_merged = SKCM_merged.dropna(subset=[
    'MMP9',
    'ajcc_nodes_pathologic_pn',
    'ajcc_metastasis_pathologic_pm',
    'ajcc_pathologic_tumor_stage',
    'OS',
    'OS.time'
])

# %%
# --- CLINICAL ASSOCIATIONS ---
####################################################

# Metastasis
sns.boxplot(data=SKCM_merged,
            x='ajcc_metastasis_pathologic_pm',
            y='MMP9')
plt.title("MMP9 vs Metastasis Status")
plt.show()

# Tumor stage
sns.boxplot(data=SKCM_merged,
            x='ajcc_pathologic_tumor_stage',
            y='MMP9')
plt.title("MMP9 vs Tumor Stage")
plt.xticks(rotation=45)
plt.show()

# Survival status
sns.boxplot(data=SKCM_merged,
            x='OS',
            y='MMP9')
plt.title("MMP9 vs Survival Status")
plt.show()

# Survival time
sns.scatterplot(data=SKCM_merged,
                x='OS.time',
                y='MMP9')
plt.title("MMP9 vs Survival Time")
plt.show()

# %%
# --- PCA ---
####################################################
X = SKCM_gene_data.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=X.index)
pca_df = pca_df.merge(SKCM_metadata, left_index=True, right_index=True)

print("Explained variance:", pca.explained_variance_ratio_)

# Color PCA by MMP9
pca_df['MMP9_expr'] = SKCM_gene_data.loc['MMP9']

sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='MMP9_expr', palette='viridis')
plt.title("PCA colored by MMP9 expression")
plt.show()

# %%
# --- CLUSTERING ---
####################################################
kmeans = KMeans(n_clusters=3, random_state=42)
pca_df['cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='cluster', palette='tab10')
plt.title("PCA with KMeans clusters")
plt.show()

# Compare MMP9 across clusters
pca_df['MMP9_expr'] = SKCM_gene_data.loc['MMP9']
print(pca_df.groupby('cluster')['MMP9_expr'].mean())

# %%
# --- OPTIONAL: HIGH vs LOW MMP9 ---
####################################################
SKCM_merged['MMP9_high'] = SKCM_merged['MMP9'] > SKCM_merged['MMP9'].median()

print(SKCM_merged.groupby('MMP9_high')['OS'].mean())