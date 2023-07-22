from flask import Flask, render_template, Response, request, jsonify
import cv2
import pyodbc
import detect_face
import imutils
import numpy as np
import training
from numpy import asarray
from numpy import expand_dims
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet
from keras_vggface.vggface import VGGFace
from numpy import load
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import time
import json
from itertools import groupby
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from flask_jwt_extended import create_access_token, get_jwt, get_jwt_identity, unset_jwt_cookies, jwt_required, JWTManager
from keras.models import load_model
from random import choice

app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = "SECRET_KEY_PROJECT_ATTENDED_SYSTEM"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

@app.route('/login', methods=["POST"])
def login():
    try:
        accounts = []
        user = request.json.get("user", None)
        password = request.json.get("password", None)

        conn = connection()
        cursor = conn.cursor()
        cursor.execute("SELECT ID, Name, Email, Role FROM Account WHERE Username = ? and Password = ?",user, password)
    
        for row in cursor.fetchall():
            accounts.append({"id": row[0], "name": row[1], "email": row[2], "role": row[3]})

        conn.close()
    
        if accounts != [] and len(accounts) > 0:
            account = accounts[0]
            access_token = create_access_token(identity=user)
            response = {"success": True,"access_token":access_token, "account": account}
        else:
            response = {"success": False,"msg": "Wrong user or password"}
    except Exception as e:
        response = {"success": False, "msg": str(e)}

    return response 
    

@app.route("/logout", methods=["POST"])
def logout():
    try:
        response = jsonify({"msg": "logout successful"})
        unset_jwt_cookies(response)
    except Exception as e:
        response = {"success": False, "msg": str(e)}

    return response

def connection():
    s = '.' #Your server name 
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
        global currentCamera
        currentCamera = frame
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

def loadListStudent_Attend(year,semester,maMH,group,session):
    students = []
    classID = 0
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("exec GetListStudents ?,?,?,?", year, semester, maMH, group)
    for row in cursor.fetchall():
        classID = row[0]
        students.append({"mssv": row[1], "name": row[2], "datesession": convertDateToString(row[session+2]), "session": session})
    conn.close()

    return students, classID

@app.route("/getListStudents_Attend", methods=["POST"])
def getListStudents_Attend():
    try:
        year = request.json.get("year", None)
        semester = request.json.get("semester", None)
        maMH = request.json.get("maMH", None)
        group = request.json.get("group", None)
        session = request.json.get("session", None)

        students, classID = loadListStudent_Attend(year,semester,maMH,group,session)
        
        if students != [] and len(students) > 0:
            result = {
                "success": True,
                "currentCode": classID,
                "listStudents": students
            }
        else:
            result = {
                "success": False,
                "msg": "Không tìm thấy lớp học này"
            }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def convertDateToString(date):
    result = None
    if (date):
        result = date.strftime("%d/%m/%Y")
    return result

def loadListStudent(year,semester,maMH,group):
    students = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("exec GetListStudents ?,?,?,?", year, semester, maMH, group)
    for row in cursor.fetchall():
        students.append({"mssv": row[1], "name": row[2], 
            "sessions": [ convertDateToString(row[3]), convertDateToString(row[4]), convertDateToString(row[5]), convertDateToString(row[6]), 
            convertDateToString(row[7]), convertDateToString(row[8]), convertDateToString(row[9]), convertDateToString(row[10]),
            convertDateToString(row[11]), convertDateToString(row[12]), convertDateToString(row[13]), convertDateToString(row[14]),
            convertDateToString(row[15]), convertDateToString(row[16]), convertDateToString(row[17])],
            'status': row[18]})
    conn.close()

    return students

def getSessions(year,semester,maMH,group):
    session = 0
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("Select Sessions from Class where Year = ? and Semester = ? and MaMH = ? and Nhom = ?", year, semester, maMH, group)
    for row in cursor.fetchall():
        session = row[0]
    conn.close()
    return session

@app.route("/getListStudents", methods=["POST"])
def getListStudents():
    try:
        year = request.json.get("year", None)
        semester = request.json.get("semester", None)
        maMH = request.json.get("maMH", None)
        group = request.json.get("group", None)

        students = loadListStudent(year,semester,maMH,group)
        sessions = getSessions(year,semester,maMH,group)

        if students != [] and len(students) > 0:
            result = {
                "success": True,
                "sessions": sessions,
                "listStudents": students
            }
        else:
            result = {
                "success": False,
                "msg": "Không tìm thấy lớp"
            }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def loadListClasses(year,semester,user):
    classes = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("exec GetListClass ?,?,?", year, semester, user)
    for row in cursor.fetchall():
        classes.append({"maMH": row[0], "TenMH": row[1], "Nhom": row[2],
        "sessions": [ convertDateToString(row[3]), convertDateToString(row[4]), convertDateToString(row[5]), convertDateToString(row[6]), 
            convertDateToString(row[7]), convertDateToString(row[8]), convertDateToString(row[9]), convertDateToString(row[10]),
            convertDateToString(row[11]), convertDateToString(row[12]), convertDateToString(row[13]), convertDateToString(row[14]),
            convertDateToString(row[15]), convertDateToString(row[16]), convertDateToString(row[17])],
        "status": row[18], "session": row[19]})
    conn.close()

    return classes
@app.route("/getListClass", methods=["POST"])
def getListClass():
    try:
        year = request.json.get("year", None)
        semester = request.json.get("semester", None)
        user = request.json.get("user", None)

        classes = loadListClasses(year,semester,user)

        if classes != [] and len(classes) > 0:
            result = {
                "success": True,
                "classes": classes
            }
        else:
            result = {
                "success": False,
                "msg": "Không tìm thấy lớp"
            }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def loadListStudent_Code(code):
    students = []

    conn = connection()
    cursor = conn.cursor()
    cursor.execute("Select StudentID from Attended where ClassID = ?", code)
    for row in cursor.fetchall():
        students.append(row[0])
    conn.close()

    return students

@app.route("/diemdanh/<int:code>")
def diemdanh(code):
    try:
        students = loadListStudent_Code(code)

        # Nhận dạng
        global modelDetector 
        global modelFacenet
        global modelLabelEncoder
        global modelSVC
        global currentCamera

        # Lặt hình lại cho đúng chiều
        img = cv2.flip(currentCamera, 1)
        # Thực hiện detect ảnh
        start4 = time.time()
        msg, face_array = detect_face.dectect(modelDetector, img)
        print("---Detect Face: %s seconds ---" % (time.time() - start4))
        
        if msg == '':
            start6 = time.time()
            # Thực hiện chuyển đổi mảng pixel thành mảng các vector
            embedding = training.get_embedding(modelFacenet, face_array)
            print("---Embedding: %s seconds ---" % (time.time() - start6))
            embedding = embedding.reshape(1,-1)
            in_encoder = Normalizer(norm='l2')
            trainX = in_encoder.transform(embedding)
            #Predict
            start7 = time.time()
            # Sử dụng model SVC để dự đoán
            samples = expand_dims(trainX[0], axis=0)
            yhat_class = modelSVC.predict(samples)
            yhat_prob = modelSVC.predict_proba(samples)
            print("---Predict: %s seconds ---" % (time.time() - start7))

            # Thực hiện lấy tên và phần trăm chính xác sau khi dự đoán
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = modelLabelEncoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

            # Nếu phần trăm dự đoán trên 50 thì trả về thông tin sinh viên đó
            if (class_probability > 50.0):
                # Kiểm tra sinh viên đó có phải trong lớp đang điểm danh hay không
                studentID = next((s for s in students if s == predict_names[0]),None)
                if (studentID != None):
                    result = {
                        "success": True,
                        "data": studentID
                    }
                else:
                    result = {
                        "success": False,
                        "msg": "Không tìm thấy sinh viên trong lớp."
                    }
            # Nếu phần trăm dự đoán thấp hơn thì có thể sinh viên đó không có trong database hoặc hình ảnh từ camera kém
            else:    
                result = {
                    "success": False,
                    "msg": "Không tìm thấy sinh viên này trong cơ sở dữ liệu hoặc chất lượng hình ảnh kém"
                }
        else:
            result = {
                "success": False,
                "msg": msg
            }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def updateAttened(classID, listStudents):
    conn = connection()
    cursor = conn.cursor()
    
    for student in listStudents:
        date = None
        if (student.get('datesession') != None):
            date = datetime.strptime(student.get('datesession'), "%d/%m/%Y")
        sqlUpdate = "UPDATE Attended SET Buoi" + str(student.get('session')) +" = ? Where ClassID = ? and StudentID = ?"
        cursor.execute(sqlUpdate, date, classID, student.get('mssv'))
    
    conn.commit()
    conn.close()
    return

def updateStatusAttend(classID):
    
    return

@app.route("/submitAttended", methods=["POST"])
def submitAttended():
    try:
        classID = request.json.get("classID", None)
        listStudents = request.json.get("listStudents", None)

        updateAttened(classID, listStudents)
        # updateStatusAttend(classID)

        result = {
            "success": True,
            "msg": "Submit Attended Successful!"
        }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

@app.route("/importCourse", methods=["POST"])
def importCourse():
    try:
        year = request.form.get('year')
        semester = request.form.get('semester')
        file = request.files.get('file')
        createBy = request.form.get('createBy')
        createByName = request.form.get('createByName')
        
        columnName = ['Mã số SV', 'Mã MH', 'Tên MH', 'Nhóm']

        if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            data_excel = pd.read_excel(file.read(), usecols=columnName, engine='openpyxl')  # XLSX
        elif file.content_type == 'application/vnd.ms-excel':
            data_excel = pd.read_excel(file.read(), usecols=columnName)  # XLS

        data_json_string = data_excel.to_json(orient='records')
        data = json.loads(data_json_string)
        
        conn = connection()
        cursor = conn.cursor()

        #Group By maMH, Nhom
        data_group = groupby(data, lambda item: (item["Mã MH"], item["Tên MH"], item["Nhóm"]))
        for k,g in data_group:
            # k: Includes MaMH and Group to create 1 class
            # g: list student in class
            # Create Class
            sqlInsertClass = 'INSERT INTO Class (Year, Semester, MaMH, TenMH, Nhom, Sessions, CreateBy, CreateByName)' + \
                            'VALUES(?,?,?,?,?,?,?,?)'
            cursor.execute(sqlInsertClass, year, semester, k[0], k[1], k[2], 15, createBy, createByName)
            record_id = cursor.execute('SELECT @@IDENTITY AS id;').fetchone()[0]
            cursor.commit()
            # Create Student in Class
            for item in list(g):
                sqlInsertStudentInClass = 'INSERT INTO Attended (ClassID, StudentID, Status) VALUES(?,?,?)'
                cursor.execute(sqlInsertStudentInClass, record_id, item.get('Mã số SV'), 'active')
                cursor.commit()

        conn.close()

        result = {
            "success": True,
            "msg": "Import Course Success."
        }
    except pyodbc.Error as e:
        result = {"success": False, "msg": "Import Course Failed."}
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

@app.route("/importStudents", methods=["POST"])
def importStudents():
    try:
        file = request.files.get('file')
        createBy = request.form.get('createBy')
        createByName = request.form.get('createByName')

        columnName = ['Mã số SV', 'Họ lót', 'Tên', 'Phái', 'Email']

        if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            data_excel = pd.read_excel(file.read(), usecols=columnName, engine='openpyxl')  # XLSX
        elif file.content_type == 'application/vnd.ms-excel':
            data_excel = pd.read_excel(file.read(), usecols=columnName)  # XLS

        data_json_string = data_excel.to_json(orient='records')
        data = json.loads(data_json_string)
        
        conn = connection()
        cursor = conn.cursor()

        for row in data:
            sqlInsertStudents = 'INSERT INTO Students (ID, Name, Sex, Email)' + \
                            'VALUES(?,?,?,?)'
            cursor.execute(sqlInsertStudents, row.get('Mã số SV'), row.get('Họ lót') + ' ' + row.get('Tên'), row.get('Phái'), row.get('Email'))
            
        cursor.commit()
        conn.close()
        
        result = {
            "success": True,
            "msg": "Import Students Success."
        }
    except pyodbc.Error as e:
        result = {"success": False, "msg": "Import Students Failed."}
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

@app.route("/loadCourseData", methods=["POST"])
def loadCourseData():
    try:
        year = request.json.get("year", None)
        semester = request.json.get("semester", None)
        userID = request.json.get("userID", None)
        listCourses = []
        conn = connection()
        cursor = conn.cursor()
        cursor.execute("Select MaMH, TenMH, Nhom, Sessions from Class where CreateBy = ? and Year = ? and Semester = ?", userID, year, semester)
        for row in cursor.fetchall():
            listCourses.append({"MaMH": row[0], "TenMH": row[1], "Nhom": row[2], "Sessions": row[3]})
        conn.close()

        result = {
            "success": True,
            "data": listCourses
        }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def retrainModel():
    ErrorMsg = ''
    global modelSVC
    global modelFacenet
    global modelDetector
    # Bắt đầu Embedding dữ liệu ảnh mới
    
    training.mainTraining(modelDetector,modelFacenet)
    # Fit lại model vừa mới train
    
    modelSVC = load_modelSVC()

    return ErrorMsg

@app.route("/themsinhvien")
def themsinhvien():
    result = 'thanh cong'
    ErrorMsg = retrainModel()
    return result

def load_modelSVC():
    model = SVC(kernel='linear', probability=True)
    modelPath = os.path.join(os.path.dirname(__file__),'Model/faces-embeddings.npz')
    if os.path.exists(modelPath):
        data = load('Model/faces-embeddings.npz')

        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        global modelLabelEncoder
    
        modelLabelEncoder.fit(trainy)
        trainy = modelLabelEncoder.transform(trainy)
        testy = modelLabelEncoder.transform(testy)

        model.fit(trainX, trainy)

    return model

def load_AllModel():
    mDectector = MTCNN(margin=10, select_largest=False)
    mSVC = load_modelSVC()
    mFacenet = load_model('facenet_keras.h5')#VGGFace(model='resnet50')#FaceNet()#
    return mDectector, mFacenet, mSVC

@app.route("/reviewModel")
def reviewModel():
    result = 'thanh cong'
    global modelSVC
    data = load('Model/faces-embeddings.npz')

    curtrainX, curtrainy, curtestX, curtesty = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(curtrainX)
    testX = in_encoder.transform(curtestX)

    global modelLabelEncoder
    
    modelLabelEncoder.fit(curtrainy)
    trainy = modelLabelEncoder.transform(curtrainy)
    testy = modelLabelEncoder.transform(curtesty) 

    # Danh gia
    # predict
    yhat_train = modelSVC.predict(trainX)
    yhat_test = modelSVC.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    return result

@app.route("/randomTest")
def randomTest():
    global modelSVC
    # load face embeddings
    data = load('Model/faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # test model on a random example from the test dataset
    
    for selection in range(45):
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = modelSVC.predict(samples)
        yhat_prob = modelSVC.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    return "done"

if __name__ == "__main__":
    currentCamera = []
    modelLabelEncoder = LabelEncoder()
    modelDetector, modelFacenet, modelSVC = load_AllModel()
    app.run(debug=True)