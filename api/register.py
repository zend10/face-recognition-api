from imutils import paths
import os
import pickle
import cv2
import dlib
import numpy as np
import face_recognition_models
import descriptor as dcp

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

def registerFaces(assetPath):
    imagePaths = list(paths.list_images(assetPath))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = dcp.getBox(rgb)

        if (boxes == [(0,0,0,0)]):
            return False
        
        encodings = face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    newData = {"encodings": knownEncodings, "names": knownNames}
    data = {}

    if os.path.exists('encodings.pickle') == False:
        f = open('encodings.pickle', 'w')
        f.close()

    if os.path.getsize('encodings.pickle') > 0:
        with open('encodings.pickle', 'rb') as f:
            data = pickle.load(f)
            data["encodings"] += newData["encodings"]
            data["names"] += newData["names"]
    else:
        data = newData

    with open('encodings.pickle', 'wb') as f:
        f.write(pickle.dumps(data))

    return True

def face_encodings(face_image, known_face_locations, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.
    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def _raw_face_landmarks(face_image, face_locations, model="large"):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

