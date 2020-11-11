#Task 2
import pandas as pd
data = pd.read_csv("F:\\Digij\\Internshp TSF\\Internship\\Iris.csv")
print("Data Import Successful")

x = data.iloc[:, [1, 2, 3, 4]].values
data_new = data.iloc[:,1:5]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_new)
y_kmeans = kmeans.fit_predict(x)
print("Data Training Successful")

import matplotlib.pyplot as plt
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'green', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')
plt.legend()
plt.show()
