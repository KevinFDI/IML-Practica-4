from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import AA_utils
import pandas as pd

# cargar dataset
dataset = 'calabazas.csv'
df = pd.read_csv('datasets/'+dataset)
data = df.values

NORMALIZAR = 1
if (NORMALIZAR):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

# PARAM
EPS = 0.3
MINSAM = 5

# Compute DBSCAN
db = DBSCAN(eps=EPS, min_samples=MINSAM).fit(data)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Num estimado de clusters: %d' % n_clusters_)
print('Num estimado de outliers: %d / %d total' % (n_noise_, len(data)))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))

# visualizar predicción
AA_utils.graficar_DBSCAN_2D(data, labels, db, EPS, MINSAM, n_clusters_)
