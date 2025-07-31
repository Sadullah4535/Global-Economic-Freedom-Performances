#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd

# Read data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';', header=0)
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# Filter out rows with missing Score
df = df.dropna(subset=["Score"])

# Select top 20 and bottom 20 countries by Score
top_20 = df.sort_values(by='Score', ascending=False).head(20)
bottom_20 = df.sort_values(by='Score', ascending=True).head(20).sort_values(by='Score', ascending=False)

# Combine and assign category labels
combined = pd.concat([top_20, bottom_20])
combined["Category"] = ["Top 20"] * 20 + ["Bottom 20"] * 20

# Sort for plotting
combined = combined.sort_values(by='Score', ascending=False)

# Plot
plt.figure(figsize=(12, 12))
sns.set(style="whitegrid")

ax = sns.barplot(
    data=combined,
    y="Country", x="Score", hue="Category",
    palette={"Top 20": "#00CED1", "Bottom 20": "#FF1493"},
    dodge=False
)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

# Titles and labels
ax.set_title("Top 20 and Bottom 20 Countries by Score", fontsize=16, fontweight="bold")
ax.set_xlabel("Score")
ax.set_ylabel("Country")
plt.tight_layout()
plt.show()

# Print numeric table
print(combined[["Country", "Score", "Category"]].to_string(index=False))


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')
df = df.dropna(subset=["Score"])

# Select Top 20 and Bottom 20
top_20 = df.sort_values(by='Score', ascending=False).head(20)
bottom_20 = df.sort_values(by='Score', ascending=True).head(20)

# Label categories
top_20["Category"] = "Top 20"
bottom_20["Category"] = "Bottom 20"
combined = pd.concat([top_20, bottom_20])

# Center around mean for symmetry
mean_score = df["Score"].mean()
combined["Centered_Score"] = combined["Score"] - mean_score

# Sort for plotting
combined = combined.sort_values("Centered_Score")

# Plot
plt.figure(figsize=(10, 12))
sns.set(style="whitegrid")

# Color mapping
colors = combined["Category"].map({"Top 20": "#2E8B57", "Bottom 20": "#B22222"})

# Horizontal lines
plt.hlines(y=combined["Country"], xmin=0, xmax=combined["Centered_Score"],
           color=colors, alpha=0.6, linewidth=3)

# Dots
plt.scatter(combined["Centered_Score"], combined["Country"], color="black", zorder=3)

# Annotate scores clearly with offset
offset = 0.5
for i, row in combined.iterrows():
    x = row["Centered_Score"]
    label = f"{row['Score']:.2f}"
    if x > 0:
        plt.text(x + offset, row["Country"], label, ha='left', va='center',
                 fontsize=9, fontweight='bold')
    else:
        plt.text(x - offset, row["Country"], label, ha='right', va='center',
                 fontsize=9, fontweight='bold')

# Add center line
plt.axvline(x=0, color='gray', linestyle='--')

# Titles
plt.title("Top and Bottom 20 Countries by Score (Centered on Mean)", fontsize=15, fontweight="bold")
plt.xlabel("Score Difference from Mean")
plt.ylabel("Country")
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Print numeric table
print(combined[["Country", "Score", "Category"]].sort_values(by="Score", ascending=False).to_string(index=False))


# In[ ]:





# In[ ]:





# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# List of variables (excluding 'Score')
variables = [
    'Property Rights', 'Judicial Effectiveness', 'Government Integrity',
    'Tax Burden', 'Government Spending', 'Fiscal Health', 'Business Freedom',
    'Labor Freedom', 'Monetary Freedom', 'Trade Freedom', 'Investment Freedom', 'Financial Freedom'
]

# Country column name
country_col = 'Country'

# More vivid, saturated color palette (no pastels, more neon/bright)
vivid_palette = [
    '#FF3B30',  # Bright Red
    '#5856D6',  # Vivid Indigo
    '#4CD964',  # Neon Green
    '#FF9500',  # Bright Orange
    '#007AFF',  # Vivid Blue
    '#FF2D55',  # Hot Pink / Neon Coral
    '#34C759',  # Fresh Green
    '#AF52DE',  # Bright Purple
    '#1D1D1F',  # Almost Black (for contrast)
    '#FFD60A',  # Bright Yellow
]

# Plot settings
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 18))
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]

    # Select top 10 countries for the current variable
    top10 = df.sort_values(by=var, ascending=False).head(10)

    # Print top 10 countries to console
    print(f"\nTop 10 countries by {var}:")
    print(top10[[country_col, var]].to_string(index=False))

    # Barplot with custom vivid palette (cycle colors if more than palette size)
    palette = [vivid_palette[j % len(vivid_palette)] for j in range(len(top10))]

    barplot = sns.barplot(
        x=var,
        y=country_col,
        data=top10,
        palette=palette,
        ax=ax
    )

    # Add numeric labels on bars
    for p in barplot.patches:
        width = p.get_width()
        barplot.text(width + 0.5,  # slight offset to the right
                     p.get_y() + p.get_height() / 2,
                     f'{width:.1f}',
                     va='center',
                     fontsize=9,
                     color='black',
                     weight='bold')

    ax.set_title(f'Top 10 Countries by {var}', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

# Remove extra subplots if any
for j in range(len(variables), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[ ]:





# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# Select numerical columns (first column is country name, rest are numerical)
numeric_cols = df.columns[1:]
df_numeric = df[numeric_cols]

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Spectral', fmt=".2f", linewidths=0.5, linecolor='gray')
plt.title("Correlation Matrix - Economic Freedom Indicators", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(corr_matrix)


# In[ ]:





# In[ ]:





# ✅ Script 1: Elbow & Silhouette Method to Choose k

# In[50]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# Select numerical features (excluding 'Country' and 'Score')
features = df.columns[2:]
X = df[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method and Silhouette Scores
wcss = []
sil_scores = []
K = range(2, 11)

print("k\tWCSS (Inertia)\tSilhouette Score")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, labels)
    wcss.append(inertia)
    sil_scores.append(sil_score)
    print(f"{k}\t{inertia:.2f}\t\t{sil_score:.4f}")

# Plotting both WCSS and Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 6))

# WCSS curve - blue color
ax1.plot(K, wcss, 'bo-', label='WCSS (Inertia)')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('WCSS (Inertia)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Silhouette curve - green color
ax2 = ax1.twinx()
ax2.plot(K, sil_scores, 'go-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='green', fontsize=12)
ax2.tick_params(axis='y', labelcolor='green')

# Reference line at k=5 (red dashed)
plt.axvline(x=5, color='red', linestyle='--', linewidth=1.5, label='Reference k=5')

plt.title('Elbow Method and Silhouette Score Analysis', fontsize=14, fontweight='bold')
ax1.grid(True)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

plt.tight_layout()
plt.show()


# In[ ]:





# ✅ Script 2: 3D PCA + KMeans Clustering + Country Labels

# In[19]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects

# Load data
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# Select numeric columns and remove missing
features = df.columns[2:]
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering (k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# Cluster evaluation
cluster_counts = df_clean['cluster'].value_counts().sort_index()
sil_score = silhouette_score(X_scaled, labels)

# Colors for each cluster
colors = ['#FFD700', '#FF4500', '#32CD32', '#1E90FF', '#800080']

# 3D Plot
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with clusters
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=[colors[label] for label in labels], s=40, alpha=0.8
)

# PCA cluster centers
centers_scaled = kmeans.cluster_centers_
centers_pca = pca.transform(centers_scaled)

# Offset labels to avoid overlap
np.random.seed(42)
offsets = np.random.uniform(-0.5, 0.5, size=(len(df_clean), 3))

for i, country in enumerate(df_clean['Country']):
    cluster_id = df_clean.iloc[i]['cluster']
    center = centers_pca[cluster_id]
    x, y, z = X_pca[i]
    new_x = x + offsets[i, 0] + (x - center[0]) * 0.3
    new_y = y + offsets[i, 1] + (y - center[1]) * 0.3
    new_z = z + offsets[i, 2] + (z - center[2]) * 0.3

    txt = ax.text(
        new_x, new_y, new_z,
        country, size=7, color='black'
    )
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2.0, foreground='white'),
        path_effects.Normal()
    ])

# Axes and title
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D PCA + KMeans Clustering of Countries (k=5)')

# Legend
legend_labels = [f'Cluster {i} (n={cluster_counts[i]})' for i in range(k)]
handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
    for i in range(k)
]
ax.legend(handles, legend_labels, loc='upper right')

plt.tight_layout()
plt.show()

# Print results
print("Cluster sizes:\n", cluster_counts)
print(f"\nSilhouette Score (k={k}): {round(sil_score, 4)}")

print("\nCountries in each cluster:")
for cluster_id in range(k):
    countries = df_clean[df_clean['cluster'] == cluster_id]['Country'].tolist()
    print(f"\nCluster {cluster_id} (n={len(countries)}):")
    print(", ".join(countries))


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Use a bold and vibrant color palette
palette = sns.color_palette("tab10", n_colors=df_clean['cluster'].nunique())

num_features = len(features)
cols = 3
rows = int(np.ceil(num_features / cols))

plt.figure(figsize=(18, rows * 4))
gs = gridspec.GridSpec(rows, cols)

# Dictionary to store mean values for each feature and cluster
feature_means = {}

for idx, feature in enumerate(features):
    ax = plt.subplot(gs[idx])
    sns.boxplot(x='cluster', y=feature, data=df_clean, palette=palette, ax=ax)

    # Calculate mean values per cluster
    means = df_clean.groupby('cluster')[feature].mean()
    feature_means[feature] = means

    # Annotate mean values on the box
    for i, mean in means.items():
        ax.text(i, mean, f"{mean:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax.set_title(f"{feature} by Cluster", fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Score')

plt.tight_layout()
plt.show()

# Print number of features
print(f"Number of features: {num_features}\n")

# Print mean scores per cluster for each feature
for feature, means in feature_means.items():
    print(f"{feature} - Cluster Means:")
    print(means.round(2))
    print()


# In[ ]:





# In[ ]:





# In[47]:


from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Compute cluster means
cluster_means = df_clean.groupby('cluster')[features].mean()

# Apply Min-Max normalization
scaler = MinMaxScaler()
cluster_means_norm = pd.DataFrame(
    scaler.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_norm, annot=True, cmap='turbo', fmt='.2f')
plt.title("Normalized Average Feature Scores per Cluster")
plt.show()

# === Print numeric values ===
print("\nNormalized Average Feature Scores per Cluster (Min-Max scaled):")
print(cluster_means_norm.round(3))


# In[ ]:





# In[45]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns

# === 1. Load the data ===
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# === 2. Select numerical features (excluding 'Country' and 'Score') ===
features = df.columns[2:]
X = df[features].dropna()

# === 3. Standardize the data ===
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# === 4. Apply KMeans clustering (using k=5 as example) ===
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# === 5. Compute cluster means (original scale) ===
cluster_means = df.groupby('cluster')[features].mean()

# === 6. Normalize the means using Min-Max scaling ===
scaler_minmax = MinMaxScaler()
cluster_means_norm = pd.DataFrame(
    scaler_minmax.fit_transform(cluster_means),
    columns=features,
    index=cluster_means.index
)

# === 7. Radar Chart with numeric labels ===
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for cluster_idx, row in cluster_means_norm.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'Cluster {cluster_idx}')
    ax.fill(angles, values, alpha=0.1)

    # Add numeric labels for each point
    for angle, value in zip(angles, values):
        ax.text(angle, value + 0.03, f'{value:.2f}', fontsize=8, ha='center', va='bottom')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('Radar Chart: Cluster Profiles (Normalized)', size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# === 8. Print normalized cluster means table ===
print("\nTable: Normalized Average Feature Scores per Cluster (Min-Max Scaled):")
print(cluster_means_norm.round(3))


# ✅ English Code for Feature Importance Analysis (data.csv)

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# === Load data ===
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# === Define dependent and independent variables ===
X = df.iloc[:, 2:]  # All columns from index 2 onward, excluding 'Country' and 'Score'
y = df["Score"]

# Drop missing values to keep X and y aligned
X = X.dropna()
y = y.loc[X.index]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Use only Gradient Boosting and Random Forest models ===
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}
feature_importances = pd.DataFrame(index=X.columns)

# === Train models and collect feature importances and R2 scores ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    results[name] = score
    
    # Get feature importance
    imp = model.feature_importances_
    
    # Normalize to percentages
    imp_pct = 100 * imp / imp.sum() if imp.sum() != 0 else imp
    feature_importances[name] = imp_pct

# Transpose for plotting and add R² scores column
feature_importances = feature_importances.T
feature_importances["R2 Score"] = feature_importances.index.map(results)

# === Plotting ===
sns.set(style="whitegrid")
colors = sns.color_palette("bright")

ax = feature_importances.drop(columns=["R2 Score"]).plot(
    kind='bar',
    figsize=(12, 7),
    color=colors,
    edgecolor='black'
)

# Annotate bars with percentages
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=10, color='black', padding=3)

# Annotate R² scores above bars
for i, (model_name, row) in enumerate(feature_importances.iterrows()):
    r2 = row["R2 Score"]
    ax.text(
        i,
        ax.get_ylim()[1] * 1.05,
        f"R²={r2:.3f}",
        ha='center',
        fontsize=12,
        fontweight='bold',
        color='darkred'
    )

plt.title("📊 Feature Importances (%) by Model with R² Scores", fontsize=16, fontweight='bold')
plt.ylabel("Feature Importance (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# === Print summary ===
print("📈 R² Scores by Model:")
for model, score in results.items():
    print(f"{model:<20}: {score:.4f}")

print("\n📊 Feature Importances (%) by Model:")
print(feature_importances.drop(columns=["R2 Score"]).round(2))


# In[ ]:





# In[ ]:





# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# --- Load data ---
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# --- Feature selection ---
features = df.columns[2:]  
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# --- Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMeans clustering (k=5) ---
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = cluster_labels

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
)

# --- Models to evaluate ---
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# --- Binarize labels for AUC (even if not used for ROC plot) ---
classes = sorted(np.unique(cluster_labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

results = {}

for name, model in models.items():
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Predict probabilities
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        try:
            decision_scores = clf.decision_function(X_test)
            if decision_scores.ndim == 1:
                y_prob = np.vstack([1 - decision_scores, decision_scores]).T
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            y_prob = np.zeros((X_test.shape[0], n_classes))

    # Fill missing classes
    if y_prob.shape[1] < n_classes:
        proba_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for cls in classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        proba_df = proba_df[classes]
        y_prob = proba_df.values

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = np.nan

    results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "roc_auc": roc_auc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# --- Confusion matrices ---
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.ravel()
cm_palettes = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'coolwarm']

for idx, (name, result) in enumerate(results.items()):
    print(f"\n{name} Confusion Matrix:\n{result['conf_matrix']}\n")
    sns.heatmap(
        result["conf_matrix"], annot=True, fmt='d', cmap=cm_palettes[idx % len(cm_palettes)],
        ax=axes[idx], cbar=False, linewidths=0.8, linecolor='black'
    )
    axes[idx].set_title(f"{name} - Confusion Matrix")
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

# Clean up unused axes
for j in range(len(results), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# --- Performance Summary ---
metrics_summary = {}

for name, result in results.items():
    report = result["report"]
    accuracy = report.get("accuracy", np.nan)
    precision = report.get("macro avg", {}).get("precision", np.nan)
    recall = report.get("macro avg", {}).get("recall", np.nan)
    f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    roc_auc = result.get("roc_auc", np.nan)

    metrics_summary[name] = {
        "Accuracy": round(accuracy, 3),
        "Precision (macro avg)": round(precision, 3),
        "Recall (macro avg)": round(recall, 3),
        "F1-score (macro avg)": round(f1, 3),
        "ROC AUC (macro)": round(roc_auc, 3)
    }

metrics_df = pd.DataFrame(metrics_summary).T

print("\n--- Classification Metrics Summary ---\n")
print(metrics_df)


# In[ ]:





# In[ ]:





# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# --- Load data ---
df = pd.read_csv("data.csv", encoding="ISO-8859-9", sep=';')

# --- Feature selection ---
features = df.columns[2:]  # Select numeric columns except 'Country' and 'Score'
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# --- Standardize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMeans clustering ---
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.3, random_state=42, stratify=labels
)

# --- Classification models ---
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# --- Binarize labels for multiclass ROC AUC calculation ---
classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

results = {}
palette_list = sns.color_palette("bright", n_colors=len(models))
cm_palettes = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'coolwarm']

# --- Train models, predict, and save results ---
for idx, (name, model) in enumerate(models.items()):
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Get predicted probabilities for ROC AUC calculation
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        try:
            decision_scores = clf.decision_function(X_test)
            if decision_scores.ndim == 1:
                y_prob = np.vstack([1 - decision_scores, decision_scores]).T
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            y_prob = np.zeros((X_test.shape[0], n_classes))

    # Ensure probability matrix covers all classes
    if y_prob.shape[1] < n_classes:
        proba_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for cls in classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        proba_df = proba_df[classes]
        y_prob = proba_df.values

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_macro = np.nan

    results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc_macro,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# --- Plot ROC curves ---
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    y_prob = result["y_prob"]

    fpr = dict()
    tpr = dict()
    roc_auc_class = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_class[i] = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        ax.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc_class[i]:.3f})')

        ax.text(
            x=fpr[i][-1], y=tpr[i][-1], s=f'{roc_auc_class[i]:.3f}',
            fontsize=8, color=ax.get_lines()[-1].get_color(),
            verticalalignment='bottom', horizontalalignment='right'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} - Macro AUC: {result["roc_auc"]:.3f}')
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)

    print(f"\n{name} - Class-wise AUC values:")
    for cls, auc_val in roc_auc_class.items():
        print(f"  Class {cls}: AUC = {auc_val:.4f}")

plt.tight_layout()
plt.show()

# --- Prepare performance metrics summary ---
metrics_summary = {}
for name, result in results.items():
    report = result["report"]
    accuracy = report.get("accuracy", np.nan)
    precision = report.get("macro avg", {}).get("precision", np.nan)
    recall = report.get("macro avg", {}).get("recall", np.nan)
    f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    roc_auc = result.get("roc_auc", np.nan)

    metrics_summary[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC AUC": roc_auc
    }

# --- Print performance metrics summary nicely ---
print("\n--- Performance Metrics Summary ---\n")
for model_name, metrics in metrics_summary.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:17}: {value:.3f}")
    print()

# --- Plot performance metrics bar chart ---
import matplotlib.ticker as mtick

models_list = list(metrics_summary.keys())
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
x = np.arange(len(models_list))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(14, 8))

for i, metric in enumerate(metrics):
    values = [metrics_summary[model][metric] for model in models_list]
    positions = x + i * bar_width
    bars = ax.bar(positions, values, bar_width, label=metric)

    # Annotate bars with numeric values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
ax.set_xticklabels(models_list, rotation=30, ha='right', fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("Classification Performance Metrics Comparison", fontsize=16, weight='bold')
ax.legend(title="Metrics", fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




