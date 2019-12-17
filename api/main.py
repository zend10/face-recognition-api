from flask import Flask, request, jsonify

import base64, os, shutil, re, random
import register as reg
import identify as id


app = Flask(__name__)
secret_key = open('secret.txt', 'r').read()

@app.route("/")
def hello():
    return 'Hello, World!'


@app.route("/register", methods=['GET', 'POST'])
def register():
    if (check_secret_key(request.form.get('secretkey')) == False):
        return get_failed_response('Not allowed.')

    # Validate username
    username = request.form.get('username')
    if username == None or re.search("^[a-zA-Z0-9]+$", username) == None:
        return get_failed_response('Username is invalid.')

    # Validate images
    images = request.form.getlist('image')
    if images == None or len(images) == 0:
        return get_failed_response('Image is required.')

    # Convert base64 to image
    counter = 0
    image_data = []
    for item in images:
        try:
            image_data.append(base64.b64decode(images[counter]))
            counter += 1
        except:
            return get_failed_response('Invalid image.')

    # Remove folder if already exists
    path = 'asset/' + request.form.get('username') + '/'

    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

    # Write image to folder
    counter = 0
    for item in image_data:
        with open(path + str(counter) + '.jpg', 'wb') as f:
            f.write(item)
            counter += 1

    # machine learning here
    if reg.register_faces(path) == False:
        return get_failed_response('Please retake your image.')

    return get_successful_response('Images are registered.')


@app.route("/identify", methods=['GET', 'POST'])
def identify():
    if (check_secret_key(request.form.get('secretkey')) == False):
        return get_failed_response('Not allowed.')

    # Validate image
    image = request.form.get('image')
    if image == None:
        return get_failed_response('Image is required.')

    try:
        image_data = base64.b64decode(image)
    except:
        return get_failed_response('Invalid image.')

    # Generate random filename for the image
    path = 'temp/' + str(random.randint(10000000, 99999999)) + '.jpg'

    with open(path, 'wb') as f:
        f.write(image_data)

    # Machine learning here
    names = id.identifying_faces(path)

    # Delete image after machine learning
    os.remove(path)

    # Change this to return names from the result of identifying
    return get_successful_response('Images are identified.', names)

@app.route("/registerIntoGroup", methods=['GET', 'POST'])
def register_into_group():
    if (check_secret_key(request.form.get('secretkey')) == False):
        return get_failed_response('Not allowed.')

    # Validate username
    username = request.form.get('username')
    if username == None or re.search("^[a-zA-Z0-9]+$", username) == None:
        return get_failed_response('Username is invalid.')

    #validate group
    group = request.form.get('group')
    if group == None or re.search("^[a-zA-Z0-9]+$", group) == None:
        return get_failed_response('Group is invalid.')

    # Validate images
    images = request.form.getlist('image')
    if images == None or len(images) == 0:
        return get_failed_response('Image is required.')

    # Convert base64 to image
    counter = 0
    image_data = []
    for item in images:
        try:
            image_data.append(base64.b64decode(images[counter]))
            counter += 1
        except:
            return get_failed_response('Invalid image.')

    # Remove folder if already exists
    path = 'asset' + os.path.sep + group + os.path.sep + username + os.path.sep

    if os.path.exists(path):
        shutil.rmtree(path)

    if os.path.exists('asset' + os.path.sep + group + os.path.sep) == False:
        os.mkdir('asset' + os.path.sep + group + os.path.sep)

    os.mkdir(path)

    # Write image to folder
    counter = 0
    for item in image_data:
        with open(path + str(counter) + '.jpg', 'wb') as f:
            f.write(item)
            counter += 1

    # machine learning here
    if reg.register_faces_into_group(path) == False:
        return get_failed_response('Please retake your image.')

    return get_successful_response('Images are registered.')

@app.route("/identifyFromGroup", methods=['GET', 'POST'])
def identify_from_group():
    if (check_secret_key(request.form.get('secretkey')) == False):
        return get_failed_response('Not allowed.')

    #validate group
    group = request.form.get('group')
    if group == None or re.search("^[a-zA-Z0-9]+$", group) == None:
        return get_failed_response('Group is invalid.')

    if os.path.exists('asset' + os.path.sep + group) == False:
        return get_failed_response('Group is not registered.')

    # Validate image
    image = request.form.get('image')
    if image == None:
        return get_failed_response('Image is required.')

    try:
        image_data = base64.b64decode(image)
    except:
        return get_failed_response('Invalid image.')

    # Generate random filename for the image
    path = 'temp/' + str(random.randint(10000000, 99999999)) + '.jpg'

    with open(path, 'wb') as f:
        f.write(image_data)

    # Machine learning here
    names = id.identifying_faces_from_group(path, group)

    # Delete image after machine learning
    os.remove(path)

    # Change this to return names from the result of identifying
    return get_successful_response('Images are identified.', names)

def get_failed_response(msg):
    return jsonify(
        success = False,
        message = msg,
        data = None
    )

def get_successful_response(msg, dat = None):
    return jsonify(
        success = True,
        message = msg,
        data = dat
    )

def check_secret_key(key):
    if (key == None or key != secret_key):
        return False
    else:
        return True

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
