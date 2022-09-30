import matplotlib.pyplot as plt
import AA_utils
from sklearn.mixture import GaussianMixture
import pandas as pd
plt.close('all')

# cargar dataset
df = pd.read_csv('datasets/calabazas2.csv')
data = df.values

# curvas
AA_utils.graficar_curva_elbow(data, 20, GMM=1)

# fit a Gaussian Mixture Model
k_optimo = 3
modelo = GaussianMixture(n_components=k_optimo,
                         verbose=True, covariance_type='full')
modelo.fit(data)
print('Score del modelo: ', modelo.score(data))

# graficar GMM (2D)
AA_utils.graficar_indice_silhouette_k(data, k_optimo)
AA_utils.graficar_GMM(data, modelo, probs=False, labels=False)
