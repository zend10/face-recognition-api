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

def identifying_faces(asset_path):
    data = pickle.loads(open('encodings.pickle', 'rb').read())
    image = cv2.imread(asset_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = dcp.get_box(rgb)
    encodings = face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    return names

def identifying_faces_from_group(asset_path, group, username, exclude = False):
    path_name = 'asset' + os.path.sep + group + os.path.sep + 'encodings.pickle'
    data = pickle.loads(open(path_name, 'rb').read())
    image = cv2.imread(asset_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = dcp.get_box(rgb)
    
    if boxes == []:
        return ["Unknown"]

    encodings = face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = compare_faces(data["encodings"], encoding)
        name = "Unknown"

        counts = {}
        totalDistance = {}
        for (index, distance) in enumerate(matches):
            if distance <= 0.4:
                name = data["names"][index]
                counts[name] = counts.get(name, 0) + 1
                totalDistance[name] = totalDistance.get(name, 0) + distance

        if username != None and exclude == False:
            if counts.get(username) != None and counts.get(username) > 0:
                return [username]
            else:
                return ["Unknown"]

        if username != None and exclude == True:
            if counts.get(username) != None and counts.get(username) > 0:
                name = "Unknown"
                del counts[username]
                del totalDistance[username]

        highestCount = 0
        for attr, value in counts.items():
            if value > highestCount:
                highestCount = value
                

        candidates = {}
        for attr, value in counts.items():
            if value == highestCount:
                candidates[attr] = totalDistance[attr]

        nearestDistance = 1
        for attr, value in candidates.items():
            distance = value / counts[attr]
            if distance < nearestDistance:
                name = attr
                nearestDistance = distance

        print(totalDistance)
        print(counts)
        print(highestCount)
        print(candidates)
        print(name)
        print(nearestDistance)
        names.append(name)

    return names


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

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    # return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    return list(face_distance(known_face_encodings, face_encoding_to_check))

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    dist = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return dist
