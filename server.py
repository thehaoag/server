from flask import Flask,render_template,Response
import cv2
import pyodbc
import detect_face
import imutils
import training
from numpy import asarray
from numpy import expand_dims
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet
from numpy import load
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import time
import os

app = Flask(__name__)

def connection():
    s = 'DESKTOP-1H1ENH8\SQLEXPRESS' #Your server name 
    d = 'DoAn' 
    u = '' #Your login
    p = '' #Your login password
    #cstr = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+s+';DATABASE='+d+';UID='+u+';PWD='+ p
    cstr = 'DRIVER={SQL Server};SERVER='+s+';DATABASE='+d+';Trusted_Connection=yes'
    conn = pyodbc.connect(cstr)
    return conn

def generate_frames():
    result = []
    
    camera = cv2.VideoCapture(0)  

    while True:
        #read camera frame        
        success,frame = camera.read()
        if not success:
            break
        
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)

        cv2.imwrite('camera.jpg', frame)

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('camera.jpg', 'rb').read() + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def loadListStudent_Code(code):
    students = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("SELECT s.MSSV, s.Name FROM Mapping m join Class c on c.ID = m.ClassID join Students s on s.ID = m.StudentID WHERE c.Code = ?",code)
    for row in cursor.fetchall():
        students.append({"mssv": row[0], "name": row[1]})
    conn.close()

    return students

@app.route("/getListStudents/<string:code>")
def getListStudents(code):
    
    students = loadListStudent_Code(code)

    count = len(students)
    #paging = [students[i:i+per_page] for i in range(0, len(students), per_page)]

    if students != [] and count > 0:
        result = {
            "success": True,
            "currentCode": code,
            "total": count,
            "listStudents": students
        }
    else:
        result = {
            "success": False,
            "msg": "Không tìm thấy mã lớp " + code
        }

    return result

@app.route("/diemdanh/<string:code>")
def diemdanh(code):

    start1 = time.time()
    students = loadListStudent_Code(code)
    print("---Load List Student: %s seconds ---" % (time.time() - start1))

    # Nhận dạng
    start2 = time.time()
    global modelDetector 
    global modelFacenet
    global modelLabelEncoder
    global modelSVC
    print("---Load Global model: %s seconds ---" % (time.time() - start2))

    required_size=(160, 160)

    start3 = time.time()
    realPath = os.path.dirname(__file__)
    img = cv2.imread(os.path.join(realPath,"camera.jpg"))
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("---Read Image: %s seconds ---" % (time.time() - start3))
    start4 = time.time()
    bounding_box, probs = modelDetector.detect(img)
    print("---Detect Face: %s seconds ---" % (time.time() - start4))

    if bounding_box is not None:
        faces_found = bounding_box.shape[0]

        if faces_found == 1:
            box = bounding_box[0]
            face = img[int(box[1]): int(box[3]),int(box[0]): int(box[2])]
            
            start5 = time.time()
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            print("---Convert Pixel: %s seconds ---" % (time.time() - start5))

            start6 = time.time()
            embedding = training.get_embedding(modelFacenet, face_array)
            print("---Embedding: %s seconds ---" % (time.time() - start6))

            #Predict
            start7 = time.time()
            samples = expand_dims(embedding, axis=0)
            yhat_class = modelSVC.predict(samples)
            yhat_prob = modelSVC.predict_proba(samples)
            print("---Predict: %s seconds ---" % (time.time() - start7))

            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = modelLabelEncoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

            if (class_probability > 50.0):
                currentStudent = next((s for s in students if s['mssv'] == predict_names[0]),None)
                if (currentStudent != None):
                    result = {
                        "success": True,
                        "msg": "",
                        "data": currentStudent
                    }
                else:
                    result = {
                        "success": False,
                        "msg": "Không tìm thấy sinh viên trong lớp "+ code
                    }
            else:    
                result = {
                    "success": False,
                    "msg": "Không tìm thấy sinh viên trong lớp " + code
                }
        else:
            result = {
                "success": False,
                "msg": "Có nhiều hơn 1 khuôn mặt trên camera"
            }
    else:
        result = {
            "success": False,
            "msg": "Không tìm thấy khuôn mặt trên camera"
        }

    return result

def load_modelSVC():
    data = load('Model/faces-dataset.npz')

    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    global modelLabelEncoder
    
    modelLabelEncoder.fit(trainy)
    trainy = modelLabelEncoder.transform(trainy)
    testy = modelLabelEncoder.transform(testy)

    modelCache = SVC(kernel='linear', probability=True)
    modelCache.fit(trainX, trainy)

    return modelCache


if __name__ == "__main__":

    modelDetector = MTCNN()
    
    modelFacenet = FaceNet()
    modelLabelEncoder = LabelEncoder()
    modelSVC = load_modelSVC()
    
    app.run(debug=True)