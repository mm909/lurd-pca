import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import Bar
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA

# Inorder to make this repository public
# the data folder has been removed

# file = "data/biglurd.txt"
# featureCount = 4000
# samples = 9452
file = "data/lurd.txt"
featureCount = 2000
samples = 1531

n_components = 500

print("Opening and reading data file...")
with open(file) as lurdFile:
    text = lurdFile.read()
    text = text.split('\n')
    labels = []
    features = []
    with Bar('Reading', max=samples) as bar:
        for LableIndex in range(samples):
            startIndex = LableIndex * featureCount + LableIndex
            labels.append(text[startIndex])
            features.append([])
            for valueIndex in range(featureCount):
                features[LableIndex].append(float(text[startIndex+valueIndex+1]))
                pass
            bar.next()
            pass


print("Preforming PCA...")
pca = PCA(n_components = n_components)
pca.fit(features)
ReducedData = pca.transform(features)
RestoredData = pca.inverse_transform(ReducedData)
# tsne = TSNE(n_components=3, verbose=1, perplexity=100, n_iter=1000)
# tsne_results = tsne.fit_transform(ReducedData)

explained = np.cumsum(pca.explained_variance_ratio_)
formatedExplained = math.floor(explained[len(explained)-1]*10000)/100
print("With", n_components, "Componets", str(formatedExplained) + "% of the variance is explained.")

# colors = copy.deepcopy(labels)
# for n, i in enumerate(colors):
#       if colors[n] == 'left':
#           colors[n] = '#ff6666'
#       if colors[n] == 'right':
#           colors[n] = '#55ff55'
#       if colors[n] == 'up':
#           colors[n] = '#5555ff'
#       if colors[n] == 'down':
#           colors[n] = '#dddd00'

# Graph data in 2d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ReducedData[:, 0], ReducedData[:, 1], ReducedData[:, 2], c = colors)
# ax.set_axis_off()
# plt.show()

# Graph data in 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c = colors)
# ax.set_axis_off()
# plt.show()


print("Creating reduced" + "-" + str(featureCount) + "-" + str(n_components) + "-" +str(formatedExplained) + ".txt")
with open("reduced" + "-" + str(featureCount) + "-" + str(n_components) + "-" + str(formatedExplained) + ".txt",'w') as lurdFile:
    with Bar('WritingTest', max=samples) as bar:
        for index, sample in enumerate(labels):
            # lurdFile.write(sample+"\n")
            featureStr = sample+"\n"
            for feature in ReducedData[index]:
                featureStr += str(feature)+"\n"
                pass
            lurdFile.write(featureStr)
            bar.next()
            pass

print("Creating restored" + "-" + str(featureCount) + "-" + str(n_components) +  "-" + str(featureCount) +  "-" + str(formatedExplained) + ".txt")
with open("restored" + "-" + str(featureCount) + "-" + str(n_components) +  "-" + str(featureCount) +  "-" + str(formatedExplained) + ".txt",'w') as lurdFile:
    with Bar('Writing', max=samples) as bar:
        for index, sample in enumerate(labels):
            # lurdFile.write(sample+"\n")
            featureStr = sample+"\n"
            for feature in RestoredData[index]:
                featureStr += str(feature)+"\n"
                pass
            lurdFile.write(featureStr)
            bar.next()
            pass

# 37817453 lines in biglurd
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
