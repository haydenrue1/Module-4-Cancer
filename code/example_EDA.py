# Statement of AI usage:
# ChatGPT was used to troubleshoot PCA matching and help structure analysis code.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score


# Load expression data and clinical metadata
data = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

metadata = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

# Truncate IDs to 12 chars to match between data columns and metadata index
data_ids = data.columns.str[:12]
meta_ids = metadata.index.str[:12]

data_map = dict(zip(data_ids, data.columns))
meta_map = dict(zip(meta_ids, metadata.index))

common_ids = set(data_ids).intersection(meta_ids)

data = data[[data_map[i] for i in common_ids]]
metadata = metadata.loc[[meta_map[i] for i in common_ids]]

# Keep only melanoma (SKCM) samples
metadata = metadata[metadata['cancer_type'] == 'SKCM']
data = data[metadata.index]

# Select top 150 most variable genes and always include MMP9
gene_var = data.var(axis=1)
top_genes = gene_var.sort_values(ascending=False).head(150).index
genes = list(set(top_genes).union({'MMP9'}))

gene_data = data.loc[genes]

# Merge expression and metadata into one dataframe
df = gene_data.T.merge(metadata, left_index=True, right_index=True)

# Drop samples missing key clinical variables
df = df.dropna(subset=[
    'MMP9', 'ajcc_metastasis_pathologic_pm',
    'ajcc_pathologic_tumor_stage', 'OS', 'OS.time'
])


# Boxplots to check if MMP9 expression differs across clinical groups
sns.boxplot(data=df, x='ajcc_metastasis_pathologic_pm', y='MMP9')
plt.title("MMP9 vs Metastasis")
plt.show()

sns.boxplot(data=df, x='ajcc_pathologic_tumor_stage', y='MMP9')
plt.title("MMP9 vs Tumor Stage")
plt.xticks(rotation=45)
plt.show()

sns.boxplot(data=df, x='OS', y='MMP9')
plt.title("MMP9 vs Survival Status")
plt.show()

# Scatterplot to see if MMP9 tracks with survival time
sns.scatterplot(data=df, x='OS.time', y='MMP9')
plt.title("MMP9 vs Survival Time")
plt.show()


# Scale genes before PCA so no gene dominates due to variance differences
X = gene_data.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 principal components for visualization
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=X.index)
pca_df['MMP9'] = gene_data.loc['MMP9']

print("Explained variance:", pca.explained_variance_ratio_)

# Color PCA plot by MMP9 expression to see if it separates along any axis
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='MMP9')
plt.title("PCA colored by MMP9")
plt.show()


# Check which genes drive each PC
loadings = pd.DataFrame(
    pca.components_.T,
    index=gene_data.index,
    columns=['PC1', 'PC2']
)

top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(10)
top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(10)

print("\nTop 10 genes driving PC1:\n", top_pc1)
print("\nTop 10 genes driving PC2:\n", top_pc2)

print("\nMMP9 Loadings:")
print(loadings.loc['MMP9'])

top_pc1.plot(kind='bar')
plt.title("Top Genes Contributing to PC1")
plt.tight_layout()
plt.show()


# Cluster samples in PCA space using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
pca_df['cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster')
plt.title("KMeans Clusters")
plt.show()

# Check if MMP9 expression differs across clusters
print(pca_df.groupby('cluster')['MMP9'].mean())


# Build feature matrix from gene expression + PC scores
X_model = gene_data.T.loc[df.index].copy()
y = df['OS'].astype(int)

X_model['PC1'] = pca_df.loc[X_model.index, 'PC1']
X_model['PC2'] = pca_df.loc[X_model.index, 'PC2']

X_train, X_val, y_train, y_val = train_test_split(
    X_model, y, test_size=0.2, random_state=42
)

# Train random forest to classify survival status
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_val = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]

print("\nClassification Performance")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation ROC-AUC:", roc_auc_score(y_val, y_prob))

fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.show()

# Large gap between train/val error suggests overfitting
print("Train Error:", 1 - accuracy_score(y_train, y_pred_train))
print("Validation Error:", 1 - accuracy_score(y_val, y_pred_val))

# See which genes contributed most to classification
importances = pd.Series(clf.feature_importances_, index=X_model.columns)
print("Top features:\n", importances.sort_values(ascending=False).head(10))
print("MMP9 importance:", importances['MMP9'])


# Regression model to predict survival time
y_time = df['OS.time']

X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_model, y_time, test_size=0.2, random_state=42
)

# max_depth=5 to reduce overfitting
reg = RandomForestRegressor(max_depth=5, random_state=42)
reg.fit(X_train_r, y_train_r)

y_pred_train_r = reg.predict(X_train_r)
y_pred_val_r = reg.predict(X_val_r)

print("\nRegression Performance")
print("Train MSE:", mean_squared_error(y_train_r, y_pred_train_r))
print("Validation MSE:", mean_squared_error(y_val_r, y_pred_val_r))
print("Train R^2:", r2_score(y_train_r, y_pred_train_r))
print("Validation R^2:", r2_score(y_val_r, y_pred_val_r))

plt.scatter(y_val_r, y_pred_val_r)
plt.title("Predicted vs Actual Survival Time")
plt.show()


# Evaluate trained models on a held-out external test set
test_data = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TEST_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)
test_metadata = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TEST_SET_GSE62944_metadata.csv',
    index_col=0
)

# Test set IDs already match between expression and metadata, no truncation needed
common_ids = test_data.columns.intersection(test_metadata.index)

test_data = test_data[common_ids]
test_metadata = test_metadata.loc[common_ids]

# Keep only SKCM samples
test_metadata = test_metadata[test_metadata['cancer_type'] == 'SKCM']
test_data = test_data[test_metadata.index]

# Use the exact same genes selected during training
test_gene_data = test_data.loc[test_data.index.intersection(genes)]
test_gene_data = test_gene_data.reindex(genes, fill_value=0)

# Merge and drop samples missing survival label
test_df = test_gene_data.T.merge(test_metadata, left_index=True, right_index=True)
test_df = test_df.dropna(subset=['OS'])

# Pull features from merged df to keep index consistent
gene_cols = [g for g in genes if g in test_df.columns]
X_test = test_df[gene_cols].copy()
X_test = X_test.reindex(columns=X_model.drop(columns=['PC1', 'PC2']).columns, fill_value=0)
y_test = test_df['OS'].astype(int)

# Apply the same scaler and PCA fitted on training data
X_test_scaled = scaler.transform(X_test)
pcs_test = pca.transform(X_test_scaled)
X_test['PC1'] = pcs_test[:, 0]
X_test['PC2'] = pcs_test[:, 1]

y_pred_test = clf.predict(X_test)
y_prob_test = clf.predict_proba(X_test)[:, 1]

print("\nExternal Test Set Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_test))

fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
plt.plot(fpr_test, tpr_test)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (External Test Set)")
plt.savefig("plot_name.png")
plt.close()