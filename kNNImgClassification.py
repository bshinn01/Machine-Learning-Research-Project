# import packages for image preprocessing and kNN model
import numpy as np
from keras_preprocessing import image
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
import pandas as pd
import warnings
from sklearn.neighbors import KNeighborsClassifier
from builtins import range
from builtins import object

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30*30 + 1)

# My code starts here

# Function to find and load image files into an array
def load_image_files(container_path, dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    count = 0
    training_img = []
    # read all image files and remove color, then assign to training image
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            count += 1
            img = imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_pred = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
            img_pred = image.img_to_array(img_pred)
            img_pred = img_pred / 255
            training_img.append(img_pred)

    X = np.array(training_img)
    return X


# load image files
X = []
X = load_image_files("dogs-vs-cats/subset_train")
y0 = np.zeros(1500)  # number of cat photos
y1 = np.ones(1500)  # number of dog photos
y = []
y = np.concatenate((y0, y1), axis=0)

# create training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

# shape training data
num_training = X_train.shape[0]
pad = list(range(num_training))
X_train = X_train[pad]
y_train = y_train[pad]

num_test = X_test.shape[0]
pad1 = list(range(num_test))
X_test = X_test[pad1]
y_test = y_test[pad1]

num_val = X_val.shape[0]
pad2 = list(range(num_val))
X_val = X_val[pad2]
y_val = y_val[pad2]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

# define the kNN class
class kNN(object):
    def __init__(self):
        pass

    def predict_label(self, dists, k=1):
        num_val = dists.shape[0]
        y_pred = np.zeros(num_val)
        for i in range(num_val):
            closest_y = self.y_train[np.argsort(dists[i])][0:k]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances_no_loops(X)
        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt((X ** 2).sum(axis=1, keepdims=1) + (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        num_val = dists.shape[0]
        y_pred = np.zeros(num_val)
        for i in range(num_val):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])][0:k]
            closest_y = closest_y.astype(int)
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred


# to determine the best k value
lix = []
liy = []
index = 0
acc = 0
for k in range(1, 100):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    liy.append(neigh.score(X_val, y_val))
    if liy[k-1] > acc:
        acc = liy[k-1]
        index = k-1
    lix.append(k)

plt.plot(lix, liy)
plt.title("k-NN: Classification Accuracy")
plt.show()
print("max acc at k="+str(index+1)+" acc of "+str(acc))

# to determine test accuracy for kNN
neigh = KNeighborsClassifier(n_neighbors=30)
neigh.fit(X_train, y_train)
print("Test Accuracy: "+str(neigh.score(X_val, y_val)))

# classifier for predicting new images
print("Predicting custom image")
img = cv.imread("dogs-vs-cats/test1/1.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_pred = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
img_pred = image.img_to_array(img_pred)
img_pred = img_pred/255
img_pred = np.reshape(img_pred, (1, img_pred.shape[0]*img_pred.shape[1]))


classifier = kNN()
classifier.train(X_train, y_train)

# Test your implementation:
dists2 = classifier.compute_distances_no_loops(img_pred)
labels = ["Cat", "Dog"]
y_val_pred = classifier.predict_labels(dists2, k=30)
print(labels[int(y_val_pred)])
