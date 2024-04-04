import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class KMeans_Custom:
    def __init__(self, n_clusters=8, max_iter=300, random_state=42):
        '''
        Custom KMeans algorithm written specially for this problem
        :param n_clusters: Number of Clusters, initially is 8
        :param max_iter: Number of Maximum Iterations, initially is 300
        :param random_state: Random State
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None

    def fit(self, X):
        np.random.seed(self.random_state)
        self.centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each point to the nearest center
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2), axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(new_centers, self.centers): # Check for convergence
                break

            self.centers = new_centers

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2), axis=1)


def reduce_colors(image_path, k):
    # load the image and convert it to a numpy array
    image = Image.open(image_path)
    img_array = np.array(image)

    # flatten the array to be a list of pixels
    pixels = img_array.reshape(-1, img_array.shape[-1])

    # fit the k-means algorithm to the pixels
    kmeans_custom = KMeans_Custom(n_clusters=k, random_state=0)
    kmeans_custom.fit(pixels)

    # get the labels and centers of the clusters
    labels_custom = kmeans_custom.predict(pixels)
    centers_custom = kmeans_custom.centers

    # replace each pixel with its nearest center
    new_pixels_custom = centers_custom[labels_custom].reshape(img_array.shape)

    # convert the array back to an image and return it
    return Image.fromarray(np.uint8(new_pixels_custom))


# generate some random data
m = 50
X = np.vstack([np.random.randn(m, 2) + [4, 4],
               np.random.randn(m, 2) + [0, -4],
               np.random.randn(m, 2) + [-4, 4]])

# plot the data
plt.rcParams['figure.figsize'] = [20 / 2.54, 15 / 2.54]
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.show()

# Fit the k-means algorithm
kmeans = KMeans_Custom(n_clusters=3, max_iter=300)
kmeans.fit(X)

# Get the labels and centers of the clusters
labels = kmeans.predict(X)
centers = kmeans.centers

print("Labels:", labels)
print("Centers:", centers)

# plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=25, cmap='viridis')

# plot the centers of each cluster
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
plt.show()

# Elbow method
inertias = []

for i in range(1, 11):
    kmeans_custom = KMeans_Custom(n_clusters=i, max_iter=300)
    kmeans_custom.fit(X)
    inertias.append(np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - kmeans_custom.centers, axis=2), axis=1)))

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Reduce colors in an image
image_path = "3_kmeans/parrot.jpeg"
k = 10
new_image = reduce_colors(image_path, k)
new_image.show()

image_path = "3_kmeans/Rainbow.png"
k = 7
new_image = reduce_colors(image_path, k)
new_image.show()
