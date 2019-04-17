''' Hybrid Deep Learning Model for Cancer recognition
Using SOM and ANN
I use the SOM in an usupervised manner to aggragate the malignant tumors -without knowing they are malignant/benign
and than i use the ANN to be able to predict from the tumor data(radius,texture etc..)
the patiants the probabilty for having a malignant tumor
'''


#Identify the tumors with the Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('C:/Users/asaelb/Desktop/cancerda.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
Y=pd.get_dummies(y)
y=Y.loc[:,'M']

dataset.loc[:,'M']=y
dataset=dataset.drop(columns='diagnosis')
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 31, sigma = 1.0, learning_rate = 0.5,random_seed=2)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the malignant tumor
mappings = som.win_map(X)
toumors= np.concatenate((mappings[(2,4)], mappings[(8,3)]), axis = 0)#better check the seed first-has some deviations from what iv'e seen
#toumors = mappings[(8,3)]
toumors = sc.inverse_transform(toumors)

# Creating the matrix of features
patients = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_toumor = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in toumors:
        is_toumor[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
patients = sc.fit_transform(patients)

#ANN

# Importing the Keras libraries and packages
from keras import Sequential
from keras import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(patients, is_toumor, batch_size = 1, epochs = 2)

# Predicting the probabilities of having tumor
y_pred = classifier.predict(patients)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]