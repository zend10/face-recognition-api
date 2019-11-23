from flask import Flask, request, jsonify

import base64, os, shutil, re, random
import register as reg
import identify as id


app = Flask(__name__)
secretKey = open('secret.txt', 'r').read()

@app.route("/")
def hello():
    return 'Hello, World!'


@app.route("/register", methods=['GET', 'POST'])
def register():
    if (checkSecretKey(request.form.get('secretkey')) == False):
        return getFailedResponse('Not allowed.')

    # Validate username
    username = request.form.get('username')
    if username == None or re.search("^[a-zA-Z0-9]+$", username) == None:
        return getFailedResponse('Username is invalid.')

    # Validate images
    images = request.form.getlist('image')
    if images == None or len(images) == 0:
        return getFailedResponse('Image is required.')

    # Convert base64 to image
    counter = 0
    imageData = []
    for item in images:
        try:
            imageData.append(base64.b64decode(images[counter]))
            counter += 1
        except:
            return getFailedResponse('Invalid image.')

    # Remove folder if already exists
    path = 'asset/' + request.form.get('username') + '/'

    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

    # Write image to folder
    counter = 0
    for item in imageData:
        with open(path + str(counter) + '.jpg', 'wb') as f:
            f.write(item)
            counter += 1

    # machine learning here
    if reg.registerFaces(path) == False:
        return getFailedResponse('Please retake your image.')

    return getSuccessfulResponse('Images are registered.')


@app.route("/identify", methods=['GET', 'POST'])
def identify():
    if (checkSecretKey(request.form.get('secretkey')) == False):
        return getFailedResponse('Not allowed.')

    # Validate image
    image = request.form.get('image')
    if image == None:
        return getFailedResponse('Image is required.')

    try:
        imageData = base64.b64decode(image)
    except:
        return getFailedResponse('Invalid image.')

    # Generate random filename for the image
    path = 'temp/' + str(random.randint(10000000, 99999999)) + '.jpg'

    with open(path, 'wb') as f:
        f.write(imageData)

    # Machine learning here
    names = id.identifyingFaces(path)

    # Delete image after machine learning
    os.remove(path)

    # Change this to return names from the result of identifying
    return getSuccessfulResponse('Images are identified.', names)


def getFailedResponse(msg):
    return jsonify(
        success = False,
        message = msg,
        data = None
    )

def getSuccessfulResponse(msg, dat = None):
    return jsonify(
        success = True,
        message = msg,
        data = dat
    )

def checkSecretKey(key):
    if (key == None or key != secretKey):
        return False
    else:
        return True

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
