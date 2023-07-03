from flask import Flask,render_template,Response
import cv2
import pyodbc
import detect_face
import imutils
import training
from numpy import asarray
from numpy import expand_dims
from mtcnn import MTCNN
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

    students = loadListStudent_Code(code)
    print(len(students))
    # Nhận dạng
    detector = MTCNN()  
    data = load('Model/faces-dataset.npz')
    model_facenet = FaceNet()
    required_size=(160, 160)

    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    realPath = os.path.dirname(__file__)
    img = cv2.imread(os.path.join(realPath,"camera.jpg"))
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces_found = detector.detect_faces(img)

    if faces_found != []:
        box = faces_found[0]['box']
        face = img[box[1]: box[1]+box[3],box[0]: box[0]+ box[2]]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        embedding = training.get_embedding(model_facenet, face_array)
        
        #Predict
        samples = expand_dims(embedding, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
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
            "msg": "Không tìm thấy khuôn mặt trên camera"
        }

    return result

if __name__ == "__main__":
    app.run(debug=True)