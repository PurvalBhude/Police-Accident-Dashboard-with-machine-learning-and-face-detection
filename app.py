from flask import Flask, render_template, request, jsonify, Response
import pickle
import numpy as np
import gzip
import cv2
from sklearn.neighbors import KNeighborsClassifier
import os
import sklearn

# Function to find the index of the maximum value in an array
def maxindex(arr):
    max_val = -1000
    max_index = -1
    for i in range(arr.size):
        if arr[i] > max_val:
            max_val = arr[i]
            max_index = i
    return max_index

# video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('resources\haarcascade_frontalface_default.xml')

# Global variables for capturing faces
capturing_faces = False
name = ""
capture_name = ""
faces_data_capture = []
i = 0

# Create a Flask application object
app = Flask(__name__)


# File path to the compressed pickle file
input_file_path = 'model/multi_output_rf_model.pkl.gz'


# Decompress and load the model
with gzip.open(input_file_path, 'rb') as f:
    model = pickle.load(f)

# List of possible classes
severity = np.array(['Damage Only','Fatal', 'Grievous Injury', 'Simple Injury'])
district = np.array(['Bagalkot', 'Ballari', 'Belagavi City', 'Belagavi Dist', 'Bengaluru City', 'Bengaluru Dist', 'Bidar', 'Chamarajanagar', 'Chickballapura', 'Chikkamagaluru', 'Chitradurga', 'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Hubballi Dharwad City', 'K.G.F', 'Kalaburagi', 'Kalaburagi City', 'Karnataka Railways', 'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mangaluru City', 'Mysuru City', 'Mysuru Dist', 'Raichur', 'Ramanagara', 'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada', 'Vijayanagara', 'Vijayapur', 'Yadgir'])
location = np.array(['City/Town', 'Rural Areas', 'Villages settlement'])
weather = np.array(['Clear', 'Cloudy', 'Dust Storm', 'Fine', 'Flooding of Slipways/Rivulets', 'Fog / Mist', 'Hail or Sleet', 'Heavy Rain', 'Light Rain', 'Mist or Fog', 'Others', 'Snow', 'Strong Wind', 'Very Cold', 'Very Hot', 'Wind'])

@app.route('/')
def index():
    return render_template('landingpage.html')

@app.route('/predict', methods=['POST'])
def predict_accident():
    # Get data from form
    W = int(request.form['weather'])
    D = int(request.form['district'])
    NumberOfVehicles = int(request.form['numberOfVehicles'])
    Latitude = float(request.form['latitude'])
    Longitude = float(request.form['longitude'])
    accident_location = int(request.form['accident_location'])

    # One-hot encode categorical variables
    Weather = np.zeros(weather.size)
    District = np.zeros(district.size)
    Location = np.zeros(location.size)

    Weather[W] = 1
    District[D] = 1
    Location[accident_location] = 1

    # Prepare the prediction array
    predict_array = np.concatenate(([NumberOfVehicles, Latitude, Longitude], District, Location, Weather)).reshape(1, -1)

    # Make prediction
    predictions = model.predict(predict_array)
    print(predictions)
    prediction = severity[maxindex(predictions[0])]
    
    # Return the prediction as JSON
    return jsonify({'severity': prediction})

@app.route("/prediction")
def prediction():
        return render_template('predict.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/heatmap')
def heatmap():
    return render_template('accidents_heatmap.html')

@app.route('/severitymap')
def severitymap():
    return render_template('accidents_map2.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/report')
def report():
    return render_template("report.html")

def generate_face():
    global capturing_faces, capture_name, faces_data_capture, i
    detect = 0
    video1 = cv2.VideoCapture(0)
    if not video1.isOpened():
        print("Error: Could not open video.")
        return
    while True:
        ret, frame = video1.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if capturing_faces:
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data_capture) <= 100 and i % 10 == 0:
                    faces_data_capture.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data_capture)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            if len(faces_data_capture) >= 100:
                capturing_faces = False
                faces_data_capture = np.asarray(faces_data_capture)
                faces_data_capture = faces_data_capture.reshape(100, -1)
                data_dir = 'data'
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                names_path = os.path.join(data_dir, 'names.pkl')
                faces_data_path = os.path.join(data_dir, 'faces_data.pkl')
                if not os.path.exists(names_path):
                    names = [capture_name] * 100
                    with open(names_path, 'wb') as f:
                        pickle.dump(names, f)
                else:
                    with open(names_path, 'rb') as f:
                        names = pickle.load(f)
                    names += [capture_name] * 100
                    with open(names_path, 'wb') as f:
                        pickle.dump(names, f)
                if not os.path.exists(faces_data_path):
                    with open(faces_data_path, 'wb') as f:
                        pickle.dump(faces_data_capture, f)
                else:
                    with open(faces_data_path, 'rb') as f:
                        faces = pickle.load(f)
                    faces = np.append(faces, faces_data_capture, axis=0)
                    with open(faces_data_path, 'wb') as f:
                        pickle.dump(faces, f)
                capturing_faces = False
                faces_data_capture = []
                capture_name = ""
                i = 0
                detect = 1
        if detect==1:
            cv2.putText(frame, "Face ID Set Successful", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video1.release()

def generate_frames():
    verified = 0
    video2 = cv2.VideoCapture(0)
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    print('Shape of Faces matrix --> ', FACES.shape)
 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    while True:

        ret, frame = video2.read()
        if not ret:
            break
        else:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = knn.predict(resized_img)
                if output[0]==name:
                    verified=1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            if verified==1:
                cv2.putText(frame, "Face id Verified", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_face')
def video_feed_face():
    return Response(generate_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods = ['GET','POST'])
def login():
    global name
    print(request.method)
    if request.method == 'POST':
        name = request.form['login_email']
        return render_template("login.html", capturing=True, name=name)
    return render_template("login.html",capturing=False)


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global capturing_faces, capture_name
    print(request.method)
    if request.method == 'POST':
        capture_name = request.form['name']
        capturing_faces = True
        return render_template('capture.html', capturing=True , capture_name=capture_name)
    return render_template('capture.html', capturing=False)

@app.route('/signup')
def signin():
    return render_template("sign-up.html")

@app.route('/msg')
def msg():
    return render_template("msg.html")


if __name__ == '__main__':
    app.run(port=4949, host="0.0.0.0")