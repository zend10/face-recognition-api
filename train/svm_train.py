import numpy as np
import joblib as joblib
import cv2


from skimage import data, color, transform
from skimage.io import imread
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from itertools import chain
from datetime import datetime

import os
import glob
import sys
sys.path.insert(1, '../hog')
import hog


def getCurrTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    

getCurrTime()
print('Retrieving positive_patches')

# Positive image
faces = fetch_lfw_people()
positive_patches = np.array([cv2.resize(img, (48, 64), interpolation=cv2.INTER_AREA) for img in faces.images])

print('positive_patches retrieved')
getCurrTime()
print('Collecting negative images')

# Negative image
images = []

imgs_to_use = ['camera', 'text', 'coins', 'moon',
              'page', 'clock', 'immunohistochemistry',
              'chelsea', 'coffee', 'hubble_deep_field']

for name in imgs_to_use:
    img = color.rgb2gray(getattr(data, name)())
    images.append(img)


for index, filename in enumerate(glob.glob(os.path.join('images', '*.jpg'))):
    img = imread(filename)
    images.append(color.rgb2gray(img))
    
    # To limit to only 150 images
    # Comment these 2 lines to train with all images in "images" folder
    # In my very specific case, I can only afford that many images
    if (len(images) == 11):
        break

print('Negative images collected')
getCurrTime()
print('Retrieving negative_patches')

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])

print('negative_patches retrieved')
getCurrTime()
print('Extracting HOG features')

X_train = np.array([hog.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1

print('HOG features extracted')
getCurrTime()
print('Finding best estimator')

grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)

print('Found best estimator')
getCurrTime()
print('Start training')

model = grid.best_estimator_
model.fit(X_train, y_train)
joblib.dump(model, 'svm_training_data.dat')

print('Training finished')
print('Training data is saved as svm_training_data.dat')
getCurrTime()


