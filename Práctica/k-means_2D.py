import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import AA_utils

# cargar dataset
dataset = 'calabazas.csv'
df = pd.read_csv('datasets/'+dataset)

# obtener k optimo
K_MAX = 16
AA_utils.graficar_curva_elbow(df, K_MAX)
AA_utils.graficar_indice_silhouette(df, K_MAX)
k_optimo = 3

# entrenar al modelo
kmeans = KMeans(n_clusters=k_optimo)
kmeans.fit(df)

# clasificar cada patrón con los centroides
labels = kmeans.predict(df)

# centroides
centers = kmeans.cluster_centers_

# analizar del clustering
silhouette_avg = silhouette_score(df, labels)
print("K =", k_optimo, "The average silhouette_score is :", silhouette_avg)
AA_utils.graficar_indice_silhouette_k(df, k_optimo)
AA_utils.visualizar_clustering_2D(df, labels, centers)
