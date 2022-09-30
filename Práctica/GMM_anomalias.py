'''
Detección de anomalías con Gaussian Mixture Models
'''

import numpy as np
import matplotlib.pyplot as plt
import AA_utils
from sklearn.mixture import GaussianMixture
import pandas as pd
plt.close('all')

# cargar dataset
df = pd.read_csv('datasets/motores.csv')
data = df.values

# fit a Gaussian Mixture Model
k = 1
modelo = GaussianMixture(n_components=k, verbose=True, covariance_type='full')
modelo.fit(data)
print('Score de todo el modelo: ', modelo.score(data))

# graficar GMM
AA_utils.graficar_GMM(data, modelo, probs=False, labels=False)

# visualizar scores
scores = modelo.score_samples(data)
# scores= np.exp(scores) # Convertir a probabilidades

#  detectar anómalos
u_scores = np.mean(scores)
s_scores = np.std(scores)
linea_de_corte = u_scores - 4*s_scores
anomalos_ind = scores < linea_de_corte
anomalos = data[anomalos_ind]
print('Datos anómalos'), print(anomalos)

plt.figure()
plt.title('log likelihood')
plt.plot(scores, 'o')
plt.hlines(linea_de_corte, 0, data.shape[0],
           color='r', linestyle='--', linewidth=3)
plt.grid()
plt.show()
