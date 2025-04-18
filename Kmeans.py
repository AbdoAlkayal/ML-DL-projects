import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load the dataset
data_path = '/Users/mk/PycharmProjects/pythonProject1/Sample_KDD(2).csv'
data = pd.read_csv(data_path)

# Inspect dataset columns
print("Columns:", data.columns)

# Assuming the class label is in the last column
data.columns = data.columns.str.strip()
class_label = data.columns[-1]
data[class_label] = data[class_label].str.strip()

# Check for missing values and drop them
data = data.dropna()

# Dataset Splitting
normal = data[data[class_label] == 'normal']
anomaly = data[data[class_label] != 'normal']

normal_train, normal_test = train_test_split(normal, train_size=0.7, random_state=42)
anomaly_train, anomaly_test = train_test_split(anomaly, train_size=0.7, random_state=42)

train_set = pd.concat([normal_train, anomaly_train]).sample(frac=1, random_state=42)
test_set = pd.concat([normal_test, anomaly_test]).sample(frac=1, random_state=42)

# Format and display train/test set distribution
def format_table(data, label_column):
    table = data[label_column].value_counts().reset_index()
    table.columns = ['Label', 'Count']
    table['Color'] = table['Label'].apply(lambda x: 'blue' if x == 'normal' else 'red')
    table_display = tabulate(table[['Label', 'Count']], headers='keys', tablefmt='fancy_grid')
    return table_display

print("Train set distribution:")
print(format_table(train_set, class_label))
print("\nTest set distribution:")
print(format_table(test_set, class_label))

# Train a Single Perceptron
X_train = train_set.iloc[:, :-1]
y_train = (train_set[class_label] == 'normal').astype(int)
X_test = test_set.iloc[:, :-1]
y_test = (test_set[class_label] == 'normal').astype(int)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Perceptron Accuracy: {accuracy * 100:.2f}%")

# K-Means Clustering with PCA Visualization
results = {}
k_values = [2, 3, 4, 5]

# Use only numerical columns for clustering (exclude the 'class' column)
X_data = data.iloc[:, :-1]  # All columns except the class label
pca = PCA(n_components=2)
X_data_pca = pca.fit_transform(X_data)

# Custom color palette with distinct, professional colors
distinct_colors = sns.color_palette("Set1", n_colors=5)  # Set1 gives distinct, professional colors

# K-Means Clustering with PCA Visualization
results = {}
k_values = [2, 3, 4, 5]

# Use only numerical columns for clustering (exclude the 'class' column)
X_data = data.iloc[:, :-1]  # All columns except the class label
pca = PCA(n_components=2)
X_data_pca = pca.fit_transform(X_data)

# Custom color palette with distinct, professional colors
distinct_colors = sns.color_palette("Set1", n_colors=5)  # Set1 gives distinct, professional colors

for k in k_values:
    # Step 1: Fit KMeans to find centroids using k-means++ initialization
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, max_iter=300)
    kmeans.fit(X_data)  # Use numerical data only
    centroids = kmeans.cluster_centers_

    # Step 2: Calculate Euclidean distance to each centroid and assign points to the closest centroid
    def euclidean_distance(x, centroid):
        return np.linalg.norm(x - centroid)

    # Calculate distances and assign points to the nearest centroid
    distances = np.array([[euclidean_distance(x, centroid) for centroid in centroids] for x in X_data.values])
    labels = np.argmin(distances, axis=1)  # Get the index of the closest centroid

    data['Cluster'] = labels
    cluster_summary = data.groupby('Cluster')[class_label].value_counts(normalize=True).unstack().fillna(0) * 100
    results[k] = cluster_summary

    # Professional, cleaner plot
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Step 3: Scatter plot for points, colored by their assigned centroid's color
    for i in range(k):
        cluster_points = X_data_pca[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[distinct_colors[i]] * len(cluster_points), s=40, edgecolor='k', alpha=0.7, label=f'Cluster {i+1}')

    # Step 4: Plot centroids with distinct thinner markers
    for i, centroid in enumerate(centroids):
        centroid_pca = pca.transform([centroid])
        ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1], c='black', marker='X', s=200, linewidths=2, label=f'Centroid {i+1}', edgecolor='white')

    # Make plot more professional by removing unnecessary gridlines, adding finer aesthetics
    ax.set_title(f"K-Means Clustering with k={k} (PCA Reduced)", fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel("Principal Component 1", fontsize=12, family='Arial')
    ax.set_ylabel("Principal Component 2", fontsize=12, family='Arial')
    ax.legend(fontsize=10, loc='best')
    ax.grid(False)  # Remove grid lines for a cleaner look

    # Customizing the plot further for professionalism
    ax.spines['top'].set_visible(False)  # Remove top spine
    ax.spines['right'].set_visible(False)  # Remove right spine
    ax.spines['left'].set_linewidth(0.5)  # Thinner left spine
    ax.spines['bottom'].set_linewidth(0.5)  # Thinner bottom spine

    # Show plot
    plt.tight_layout()
    plt.show()

for k, summary in results.items():
    print(f"\nK={k} Clustering Results:")
    print(tabulate(summary, headers='keys', tablefmt='fancy_grid', showindex=True))

