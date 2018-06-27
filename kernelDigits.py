import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib
import pandas as pd
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


digits=load_digits()
data0=digits.data #load_digits these digits are 8*8 , 64 dimensional
pca=PCA(n_components=20)

data=pca.fit_transform(data0)
params={'bandwidth':np.logspace(-1,1)}#0.1 to 10
grid=GridSearchCV(KernelDensity(),params,n_jobs=-1)
grid.fit(data)

estimator=grid.best_estimator_
print(estimator.bandwidth)#3.56
generated_digits=pca.inverse_transform(estimator.sample(30,random_state=7))

#plot the generated digits
fig=plt.figure(figsize=(10, 10))
columns = 5
rows = 6
for i in range(1, columns*rows+1):

    fig.add_subplot(rows, columns, i)
    plt.imshow(generated_digits[i-1].reshape((8,8)))
plt.show()
