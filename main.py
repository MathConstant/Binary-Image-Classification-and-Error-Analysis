import scipy.io
import pickle
import os.path
import seaborn as sns
import numpy as np
import pandas as pd
import random


import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


# load our data file
train_data = scipy.io.loadmat('extra_32x32.mat') 

# extract the images (X) and labels (y) from the dict
X = train_data['X'] 
y = train_data['y']

# view an image (e.g. 25) and print its corresponding label
img_index = 25
# plt.imshow(X[:,:,:,img_index])
# plt.show()
# print(y[img_index])

#########################################################################

# reshape our matrices into 1D vectors and shuffle (still maintaining the index pairings)
X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]).T
y = y.reshape(y.shape[0], )
X, y = shuffle(X, y, random_state=42)  # use a seed of 42 to replicate the results of tutorial

# optional: reduce dataset to a selected size (rather than all 500k examples)
size = X.shape[0]  # change to real number to reduce size
X = X[:size, :]  # X.shape should be (num_examples,img_dimensions*colour_channels)
y = y[:size]  # y.shape should be (num_examples,)

#########################################################################

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# transform the y lables into binary even or odd labels
y_test_original = y_test.copy()
for i in range(0, len(y_test)):
    if y_test[i] % 2 == 0:
        y_test[i] = 0
    else:
        y_test[i] = 1
for i in range(0, len(y_train)):
    if y_train[i] % 2 == 0:
        y_train[i] = 0
    else:
        y_train[i] = 1
print(y_test_original[0])
print(y_test[0])

pkl_filename = "imgclass_model.pkl"
if os.path.isfile(pkl_filename):
    # Load from file
    print("opening " + pkl_filename)
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)
else:
    # define our classifier and print out specs
    clf = RandomForestClassifier()
    print(clf)

    # fit the model on training data
    clf.fit(X_train, y_train)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

# predict on unseen test data
print("Strating Preds")
preds = clf.predict(X_test)
# pred = x[:-1,:].reshape(1, -1) 	# if predicting a single example it needs reshaping

# check the accuracy of the predictive model
print("Accuracy:", accuracy_score(y_test, preds))

confusion_matrix = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()

# this is transforming the X_test data into a displayable format
X_test = X_test.T.reshape(32, 32, 3, X_test.shape[0])

f_n_img, f_p_img, f_n_label, f_p_label = [], [], [], []
for i in range(0, len(y_test)):
    if (y_test[i] == 0) and (preds[i] == 1):
        img = X_test[:,:,:,i]
        f_n_img.append(img)
        f_n_label.append(y_test_original[i])
    elif (y_test[i] == 1) and (preds[i] == 0):
        img = X_test[:, :, :, i]
        f_p_img.append(img)
        f_p_label.append(y_test_original[i])

# f_n_sample = random.sample(f_n_img, 20)
# f_p_sample = random.sample(f_p_img, 20)
f_n_sample, f_n_sample_labels = shuffle(f_n_img, f_n_label)
f_p_sample, f_p_sample_labels = shuffle(f_p_img, f_p_label)

rows, cols = 5, 4

axes=[]
fig1=plt.figure()
for a in range(rows*cols):
    b = f_n_sample[a]
    axes.append( fig1.add_subplot(rows, cols, a+1))
    subplot_title = ("Label: " + str(f_n_sample_labels[a]))
    axes[-1].set_title(subplot_title)
    axes[-1].set_axis_off()
    plt.imshow(b)
fig1.tight_layout()
plt.show()

axes=[]
fig2=plt.figure()
for a in range(rows*cols):
    b = f_p_sample[a]
    axes.append(fig2.add_subplot(rows, cols, a+1))
    subplot_title = ("Label: " + str(f_p_sample_labels[a]))
    axes[-1].set_title(subplot_title)
    axes[-1].set_axis_off()
    plt.imshow(b)
fig2.tight_layout()
plt.show()
