import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA

# Inorder to make this repository public
# the data folder has been removed
with open("data/biglurd.txt") as lurdFile:
    text = lurdFile.read()
    text = text.split('\n')
    labels = []
    features = []
    for LableIndex in range(9452):
        startIndex = LableIndex*4000+LableIndex
        labels.append(text[startIndex])
        features.append([])
        for valueIndex in range(4000):
            features[LableIndex].append(float(text[startIndex+valueIndex+1]))
            pass
        pass

n_components = 3
pca = PCA(n_components = n_components)
pca.fit(features)
ReducedData = pca.transform(features)
tsne = TSNE(n_components=3, verbose=1, perplexity=100, n_iter=1000)
tsne_results = tsne.fit_transform(ReducedData)

explained = np.cumsum(pca.explained_variance_ratio_)
formatedExplained = math.floor(explained[len(explained)-1]*10000)/100
print("With", n_components, "Componets", str(formatedExplained) + "% of the variance is explained.")

colors = labels
for n, i in enumerate(colors):
      if colors[n] == 'left':
          colors[n] = '#ff6666'
      if colors[n] == 'right':
          colors[n] = '#55ff55'
      if colors[n] == 'up':
          colors[n] = '#5555ff'
      if colors[n] == 'down':
          colors[n] = '#dddd00'

# Graph data in 2d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ReducedData[:, 0], ReducedData[:, 1], ReducedData[:, 2], c = colors)
ax.set_axis_off()
plt.show()

# Graph data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c = colors)
ax.set_axis_off()
plt.show()

RestoredData = pca.inverse_transform(ReducedData)
# Save data as csv

# With 1500 Componets 99.99% of the variance is explained.
# With 1250 Componets 99.64% of the variance is explained.
# With 1100 Componets 98.94% of the variance is explained.
# With 1000 Componets 98.11% of the variance is explained.
# With 500 Componets 83.6% of the variance is explained.
# With 250 Componets 60.82% of the variance is explained.
# With 100 Componets 34.02% of the variance is explained.
# With 50 Componets 19.88% of the variance is explained.
# With 25 Componets 11.06% of the variance is explained.
# With 3 Componets 1.58% of the variance is explained.
# With 2 Componets 1.08% of the variance is explained.
