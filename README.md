# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MANOJ MV
RegisterNumber:  212222220023
*/

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/SEC/Downloads/Mall_Customers.csv")

data.head()

data.info

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = [] #within-Cluster Sum of Square
#It is the sum of squared distance between each point & the centroid in a cluster

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
**Elbow Method**
![image](https://github.com/user-attachments/assets/706fcc76-1617-4618-863a-5df91c57b306)

**KMeans**
![image](https://github.com/user-attachments/assets/4986e5cf-57c5-4431-b461-919487d47a14)

**y_pred**
![image](https://github.com/user-attachments/assets/87b27cbc-9342-47ab-aecf-e16a31c54b63)

**Customer_Segement**
![image](https://github.com/user-attachments/assets/7d22c89e-b6a3-4142-95fd-c8396f739f56)
## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
