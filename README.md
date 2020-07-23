# face-recognition-api
Face recognition implementation with HOG, SVM, and ERT. 
For learning purpose, not to be implemented into production code.

<br>

## Credits
Files in folder hog is actually taken from scikit-image, only modified for learning purpose:

https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py

<br>

svm_train.py is modified a bit from this book:

https://github.com/jakevdp/PythonDataScienceHandbook

<br>

Folder train/images is supposed to be filled with images from here:

https://github.com/JoakimSoderberg/haarcascade-negatives

<br>

ERT implementation to detect face landmarks makes use of dlib, normally we can use dlib's implementation of HOG as well to detect face, but again, for learning purpose, we're not using dlib's HOG implementation:

https://github.com/davisking/dlib

<br>

The whole face recognition process is greatly inspired by ageitgey. In fact, plenty are copied from this repository:

https://github.com/ageitgey/face_recognition

<br>

Non-Maximum Suppression code is taken from this blog:

https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

<br>

Even better, check the whole blog for many Computer Vision resources:

https://www.pyimagesearch.com/

<br>

And many more I've probably missed. Thank you.
