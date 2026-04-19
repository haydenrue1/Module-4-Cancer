# Statement of AI usage:
# ChatGPT was used to troubleshoot errors in PCA data matching and help generate clinical data plots.

# Importing the libraries we need for this analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load in the gene expression data and the metadata
# index_col=0 means the first column is used as row the labels
data = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

metadata_df = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

# Print shapes to confirm everything loaded correctly
print("Data shape:", data.shape)
print("Metadata shape:", metadata_df.shape)


# The sample IDs in the gene data and metadata don’t perfectly match,
# so we shorten them to the first 12 characters
data_short = data.columns.str[:12]
meta_short = metadata_df.index.str[:12]

# map shortened IDs back to full IDs
data_map = dict(zip(data_short, data.columns))
meta_map = dict(zip(meta_short, metadata_df.index))

# Find which patient IDs exist in both datasets
common_ids = set(data_short).intersection(set(meta_short))
print("Matched patients:", len(common_ids))

# Use only the matched samples
data_cols_final = [data_map[i] for i in common_ids]
meta_idx_final = [meta_map[i] for i in common_ids]

data = data[data_cols_final]
metadata_df = metadata_df.loc[meta_idx_final]


# Filter down to only SKCM (melanoma) samples
cancer_type = 'SKCM'

cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
common_samples = data.columns.intersection(cancer_samples)

print("Matched SKCM samples:", len(common_samples))

SKCM_data = data[common_samples]
SKCM_metadata = metadata_df.loc[common_samples]


# Select the most variable genes since they carry the most information
# Also make sure MMP9 is included even if it’s not the most variable
gene_variance = SKCM_data.var(axis=1)
top_genes = gene_variance.sort_values(ascending=False).head(150).index

genes = list(set(top_genes).union({'MMP9'}))
SKCM_gene_data = SKCM_data.loc[genes]

print("Gene data shape:", SKCM_gene_data.shape)


# Combine gene expression data with clinical metadata into one dataframe
# Transpose is needed so samples become rows
SKCM_merged = SKCM_gene_data.T.merge(
    SKCM_metadata, left_index=True, right_index=True
)


# Remove rows with missing values for key variables we need
SKCM_merged = SKCM_merged.dropna(subset=[
    'MMP9',
    'ajcc_nodes_pathologic_pn',
    'ajcc_metastasis_pathologic_pm',
    'ajcc_pathologic_tumor_stage',
    'OS',
    'OS.time'
])


# Clinical association plots

# Compare MMP9 expression across metastasis categories
sns.boxplot(data=SKCM_merged,
            x='ajcc_metastasis_pathologic_pm',
            y='MMP9')
plt.title("MMP9 vs Metastasis Status")
plt.show()

# Compare MMP9 across tumor stages
sns.boxplot(data=SKCM_merged,
            x='ajcc_pathologic_tumor_stage',
            y='MMP9')
plt.title("MMP9 vs Tumor Stage")
plt.xticks(rotation=45)
plt.show()

# Compare MMP9 between patients who are alive vs deceased
sns.boxplot(data=SKCM_merged,
            x='OS',
            y='MMP9')
plt.title("MMP9 vs Survival Status")
plt.show()

# Look at relationship between MMP9 and survival time
sns.scatterplot(data=SKCM_merged,
                x='OS.time',
                y='MMP9')
plt.title("MMP9 vs Survival Time")
plt.show()


# PCA (Principal Component Analysis)
# This reduces the dataset into 2 dimensions to visualize patterns

X = SKCM_gene_data.T

# Standardize the data so all genes are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA and keep the first 2 components
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# Store PCA results in a dataframe
pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=X.index)
pca_df = pca_df.merge(SKCM_metadata, left_index=True, right_index=True)

# Show how much variance in each component
print("Explained variance:", pca.explained_variance_ratio_)

# Color PCA plot by MMP9 expression level
pca_df['MMP9_expr'] = SKCM_gene_data.loc['MMP9']

sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='MMP9_expr', palette='viridis')
plt.title("PCA colored by MMP9 expression")
plt.show()


# Clustering using K-Means

kmeans = KMeans(n_clusters=3, random_state=42)
pca_df['cluster'] = kmeans.fit_predict(X_scaled)

# Plot PCA with cluster labels
sns.scatterplot(data=pca_df, x='PC1', y='PC2',
                hue='cluster', palette='tab10')
plt.title("PCA with KMeans clusters")
plt.show()

# Compare average MMP9 expression across clusters
pca_df['MMP9_expr'] = SKCM_gene_data.loc['MMP9']
print(pca_df.groupby('cluster')['MMP9_expr'].mean())

# Compare survival status between high and low MMP9 groups
print(SKCM_merged.groupby('MMP9_high')['OS'].mean())