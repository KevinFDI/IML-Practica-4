import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
plt.close("all")
plt.rcParams['image.cmap'] = 'jet'
# ------------------------------------------------------


def get_random_subset(pixels, N):
    # elegir un conjunto de pixeles aleatorios
    rng = np.random.RandomState(0)
    i = rng.permutation(pixels.shape[0])[:N]
    return pixels[i, :]


def plot_pixel_list(pixels, color_map, title_fig, center_data=False, centers=False):
    # realiza un gráfico 3D de los píxeles recibidos con un vector de Nx3
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    r, g, b = list(pixels[:, 0]), list(pixels[:, 1]), list(pixels[:, 2])
    axis.scatter(r, g, b, c=color_map, marker="o")
    axis.set(xlabel='Red', ylabel='Green', zlabel='Blue',
             xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title=title_fig)
    if (centers):
        axis.scatter(center_data[:, 0], center_data[:, 1],
                     center_data[:, 2], marker='*', c='red', s=600, alpha=0.8)

    plt.show()


# ------------------------------------------------------
k = 16

# cargar imagen
imagen = 'Hornocal.jpg'

img = io.imread('datasets/'+imagen)
img = np.array(img, dtype=np.float64) / 255

# visualizar imagen
plt.imshow(img)
plt.title('imagen original')

# convertir a lista de pixeles
pixels = img.reshape(img.shape[0]*img.shape[1], 3)

# datos
print('--Número de pixels  = ', pixels.shape[0])
print('--Número de colores = ', 2**24)

# Visualizar pixeles.  obtener un subset aleatorio por cuestiones de performance
pixels_subset_cant = 5000
pixels_subset = get_random_subset(pixels, pixels_subset_cant)
plot_pixel_list(pixels_subset, pixels_subset, 'pixeles originales')
print('--Número de pixeles usados para plotear y entrenar  = ', pixels_subset_cant)
print('--Número nuevos colores (K)  = ', k)

# K-means
# aplicar K-means a los píxeles
modelo = KMeans(n_clusters=k)
modelo.fit(pixels_subset)
# clasificar cada patrón (pixels) con los centroides
modelo_labels = modelo.predict(pixels_subset)

# Visualizar centros y cuantización
# visualizar resultado de K-means
pix_centers = modelo.cluster_centers_
plot_pixel_list(pixels_subset, modelo_labels,
                'resultado de K-means', centers=True, center_data=pix_centers)

# recolorear la imagen completa
new_colors = modelo.cluster_centers_[modelo.predict(pixels)]
img_recolored = new_colors.reshape(img.shape)
fig = plt.figure()
plt.title('imagen recoloreada')
plt.imshow(img_recolored)
plt.show()
