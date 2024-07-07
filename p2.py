#import lib

import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv("ab_march24.csv")
print(data)

#features
features = data[["A", "B"]]

#model
model = KMeans(n_clusters=3, random_state = 3)
res = model.fit_predict(features)
data["clusters"] = res
print(data)

#centroid
print(model.cluster_centers_)
ca = model.cluster_centers_[0]
cb = model.cluster_centers_[1]
cc = model.cluster_centers_[2]

print(ca)
print(cb)
print(cc)

#internal working
da = data[data.clusters==0]
db = data[data.clusters==1]
dc = data[data.clusters==2]

plt.figure(figsize = (12, 5))
plt.scatter(da["A"], da["B"], color="red", label="Cluster A")
plt.scatter(db["A"], db["B"], color="green", label="Cluster B")
plt.scatter(dc["A"], dc["B"], color="blue", label="Cluster C")
plt.scatter(ca[0], ca[1], marker="x", color="black", label="ca --> "+ str(ca))
plt.scatter(cb[0], cb[1], marker="x", color="brown", label="cb --> "+ str(cb))
plt.scatter(cc[0], cc[1], marker="x", color="purple", label="cc --> "+ str(cc))
plt.legend()
plt.show()

#predict
x = float(input("Enter value of a "))
y = float(input("Enter value of b "))

d = [[x, y]]
ans = model.predict(d)

if ans == 0:
	print("cluster a")
elif ans == 1:
	print("cluster b")
else :
	print("cluster c")