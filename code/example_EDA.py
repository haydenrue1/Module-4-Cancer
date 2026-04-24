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



# Load and align data


data = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

metadata = pd.read_csv(
    '/Users/haydenrue/Desktop/Comp BME/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

# Match patient IDs between datasets
data_ids = data.columns.str[:12]
meta_ids = metadata.index.str[:12]

data_map = dict(zip(data_ids, data.columns))
meta_map = dict(zip(meta_ids, metadata.index))

common_ids = set(data_ids).intersection(meta_ids)

data = data[[data_map[i] for i in common_ids]]
metadata = metadata.loc[[meta_map[i] for i in common_ids]]

# Filter melanoma samples
metadata = metadata[metadata['cancer_type'] == 'SKCM']
data = data[metadata.index]

# Select most variable genes and include MMP9
gene_var = data.var(axis=1)
top_genes = gene_var.sort_values(ascending=False).head(150).index
genes = list(set(top_genes).union({'MMP9'}))

gene_data = data.loc[genes]

# Merge gene + clinical data
df = gene_data.T.merge(metadata, left_index=True, right_index=True)

# Drop missing values
df = df.dropna(subset=[
    'MMP9', 'ajcc_metastasis_pathologic_pm',
    'ajcc_pathologic_tumor_stage', 'OS', 'OS.time'
])



# Clinical association plots


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

sns.scatterplot(data=df, x='OS.time', y='MMP9')
plt.title("MMP9 vs Survival Time")
plt.show()


# PCA


X = gene_data.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=X.index)
pca_df['MMP9'] = gene_data.loc['MMP9']

print("Explained variance:", pca.explained_variance_ratio_)

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='MMP9')
plt.title("PCA colored by MMP9")
plt.show()



# KMeans clustering


kmeans = KMeans(n_clusters=3, random_state=42)
pca_df['cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster')
plt.title("KMeans Clusters")
plt.show()

print(pca_df.groupby('cluster')['MMP9'].mean())


# PCA loadings (which genes drive variation)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=genes
)

print("Top PC1 genes:\n", loadings['PC1'].sort_values(ascending=False).head(10))
print("MMP9 loading:\n", loadings.loc['MMP9'])



# Classification model (survival status)


X_model = gene_data.T.loc[df.index].copy()
y = df['OS'].astype(int)

# Add PCA features
X_model['PC1'] = pca_df.loc[X_model.index, 'PC1']
X_model['PC2'] = pca_df.loc[X_model.index, 'PC2']

X_train, X_val, y_train, y_val = train_test_split(
    X_model, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_val = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)[:, 1]

print("\nClassification Performance")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Validation ROC-AUC:", roc_auc_score(y_val, y_prob))

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# Error
print("Train Error:", 1 - accuracy_score(y_train, y_pred_train))
print("Validation Error:", 1 - accuracy_score(y_val, y_pred_val))

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X_model.columns)
print("Top features:\n", importances.sort_values(ascending=False).head(10))
print("MMP9 importance:", importances['MMP9'])


# Regression model (survival time)


y_time = df['OS.time']

X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_model, y_time, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(max_depth=5, random_state=42)
reg.fit(X_train_r, y_train_r)

y_pred_train_r = reg.predict(X_train_r)
y_pred_val_r = reg.predict(X_val_r)

print("\nRegression Performance")
print("Train MSE:", mean_squared_error(y_train_r, y_pred_train_r))
print("Validation MSE:", mean_squared_error(y_val_r, y_pred_val_r))
print("Train R^2:", r2_score(y_train_r, y_pred_train_r))
print("Validation R^2:", r2_score(y_val_r, y_pred_val_r))

# Feature importance (regression)
importances_r = pd.Series(reg.feature_importances_, index=X_model.columns)
print("Top regression features:\n", importances_r.sort_values(ascending=False).head(10))
print("MMP9 importance (regression):", importances_r['MMP9'])