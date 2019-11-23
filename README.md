# face-recognition-api
Face recognition implementation with HOG, SVM, and ERT. 
For learning purpose, not to be implemented into production code.


## Credits
Files in folder hog is actually taken from scikit-image, only modified for learning purpose:
https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py

svm_train.py is modified a bit from this book:
https://github.com/jakevdp/PythonDataScienceHandbook

Folder train/images is supposed to be filled with images from here:
https://github.com/JoakimSoderberg/haarcascade-negatives

ERT implementation to detect face landmarks makes use of dlib, normally we can use dlib's implementation of HOG as well to detect face, but again, for learning purpose, we're not using dlib's HOG implementation:
https://github.com/davisking/dlib

The whole face recognition process is greatly inspired by ageitgey. In fact, plenty are copied from this repository:
https://github.com/ageitgey/face_recognition

Non-Maximum Suppression code is taken from this blog:
https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

Even better, check the whole blog for many Computer Vision resources:
https://www.pyimagesearch.com/


And many more I've probably missed. Thank you.