from flask import Flask, render_template, Response, request, jsonify
import cv2
#import pyodbc
import detect_face
#import imutils
import numpy as np
import training
from numpy import asarray
from numpy import expand_dims
from facenet_pytorch import MTCNN
# from keras_facenet import FaceNet
# from keras_vggface.vggface import VGGFace
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
# from random import choice
from flask_mail import Mail, Message
# from PIL import Image
from numpy import savez_compressed
import shutil
import zipfile
import urllib.request
import mysql.connector
from flask_cors import CORS, cross_origin

app = Flask(__name__)
api_cors_config = { "origins": ["https://app-regconize-tdt.onrender.com"], "methods": ["OPTIONS","GET","POST"]}
CORS(app, resources={ r"/*": api_cors_config})
#CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'
mail= Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = '51702014@student.tdtu.edu.vn' #securesally@gmail.com
app.config['MAIL_PASSWORD'] = 'dwvtfqzqfqufhphf' #'SA!@#456' student: 'dwvtfqzqfqufhphf' outsidetdt: 'zlaofkvvdnxnomfx'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

app.config["JWT_SECRET_KEY"] = "SECRET_KEY_PROJECT_ATTENDED_SYSTEM"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

@app.route('/login', methods=["POST"])
#@cross_origin()
def login():
    try:
        accounts = []
        user = request.json.get("user", None)
        password = request.json.get("password", None)

        conn = connection()
        cursor = conn.cursor()
        sql = "SELECT ID, Name, Email, Role FROM Account WHERE Username = %s and Password = %s"
        param = (user, password,)
        cursor.execute(sql, param)

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
    #s = 'thehaoag.mysql.pythonanywhere-services.com' #Your server name
    #d = 'DoAn'
    #u = 'sa' #Your login
    #p = 'M@st3rm1nd!@#'#'dbpassword123' #Your login password
    #cstr = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+s+';DATABASE='+d+';UID='+u+';PWD='+ p
    #cstr = 'DRIVER={SQL Server};SERVER='+s+';DATABASE='+d+';Trusted_Connection=yes'
    #os.environ["ODBCSYSINI"] = "/home/thehaoag/mysite"
    #conn = pyodbc.connect('DSN=sqlserverdatasource;Uid='+ u +';Pwd=' + p + ';Encrypt=yes;Connection Timeout=30;')
    #conn = pyodbc.connect(cstr)
    # Live
    #conn = mysql.connector.connect(host="thehaoag.mysql.pythonanywhere-services.com",user="thehaoag",password="dbpassword123",database="thehaoag$DoAn")
    # Local
    conn = mysql.connector.connect(host="localhost",user="root",password="",database="doan")
    return conn

def loadListStudent_Attend(year,semester,maMH,group,session):
    students = []
    classID = 0
    conn = connection()
    cursor = conn.cursor()
    sql = "call GetListStudents(%s,%s,%s,%s)"
    param = (year, semester, maMH, group, )
    cursor.execute(sql,param)
    for row in cursor.fetchall():
        classID = row[0]
        students.append({"mssv": row[1], "name": row[2], "datesession": row[session+2], "session": session})
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

def loadListStudent(year,semester,maMH,group):
    students = []
    conn = connection()
    cursor = conn.cursor()
    sql = "call GetListStudents(%s,%s,%s,%s)"
    param = (year, semester, maMH, group,)
    cursor.execute(sql, param)
    for row in cursor.fetchall():
        students.append({"mssv": row[1], "name": row[2],
            "sessions": [ row[3], row[4], row[5], row[6],
            row[7], row[8], row[9], row[10],
            row[11], row[12], row[13], row[14],
            row[15], row[16], row[17]],
            'status': row[18]})
    conn.close()

    return students

def getSessions(year,semester,maMH,group):
    session = 0
    conn = connection()
    cursor = conn.cursor()
    sql = "Select Sessions from Class where Year = %s and Semester = %s and MaMH = %s and Nhom = %s"
    param = (year, semester, maMH, group,)
    cursor.execute(sql, param)
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
    sql = "call GetListClass(%s,%s,%s)"
    param = (year, semester, user, )
    cursor.execute(sql, param)
    for row in cursor.fetchall():
        classes.append({"maMH": row[0], "TenMH": row[1], "Nhom": row[2],
        "sessions": [ row[3], row[4], row[5], row[6],
            row[7], row[8], row[9], row[10],
            row[11], row[12], row[13], row[14],
            row[15], row[16], row[17]],
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
    sql = "Select StudentID from Attended where ClassID = %s"
    param = (code,)
    cursor.execute(sql, param)
    for row in cursor.fetchall():
        students.append(row[0])
    conn.close()

    return students

@app.route("/diemdanh",methods=["POST"])
def diemdanh():
    try:
        realPath = os.path.dirname(__file__)

        code = request.form.get("code")
        file = request.form.get("file")

        urllib.request.urlretrieve(file, os.path.join(realPath,"image-camera.jpg"))

        image_camera = cv2.imread(os.path.join(realPath,'image-camera.jpg'))

        students = loadListStudent_Code(code)

        # Nhận dạng
        global modelDetector
        global modelFacenet
        global modelLabelEncoder
        global modelSVC

        # Lặt hình lại cho đúng chiều
        img = cv2.flip(image_camera, 1)
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

            # Nếu phần trăm dự đoán trên 65 thì trả về thông tin sinh viên đó
            if (class_probability > 65.0):
                # Kiểm tra sinh viên đó có phải trong lớp đang điểm danh hay không
                studentID = next((s for s in students if s == predict_names[0]),None)
                if (studentID != None):
                    result = {
                        "success": True,
                        "msg": f"Sinh viên {predict_names[0]} điểm danh thành công",
                        "data": studentID
                    }
                else:
                    result = {
                        "success": False,
                        "msg": "Không tìm thấy sinh viên trong lớp."
                    }
            # Nếu phần trăm dự đoán thấp hơn thì có thể sinh viên đó không có trong database hoặc hình ảnh từ camera kém
            elif (class_probability > 50.0):
                result = {
                    "success": False,
                    "msg": "Chất lượng hình ảnh kém hãy thử lại."
                }
            else:
                result = {
                    "success": False,
                    "msg": "Không tìm thấy sinh viên trong cơ sở dữ liệu khuôn mặt."
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
        date = 'V'
        if (student.get('datesession') != None):
            date = student.get('datesession')
        sqlUpdate = "UPDATE Attended SET Buoi" + str(student.get('session')) +" = %s Where ClassID = %s and StudentID = %s"
        param = (date, classID, student.get('mssv'),)
        cursor.execute(sqlUpdate, param)

    conn.commit()
    conn.close()
    return

def getClassInfoByID(classID):
    classInfo = 0
    conn = connection()
    cursor = conn.cursor()
    sql = "Select * from Class where ID = %s"
    param = (classID,)
    cursor.execute(sql, param)
    for row in cursor.fetchall():
        classInfo = { "Year": row[1], "Semester": row[2], "MaMH": row[3], "TenMH": row[4], "Nhom": row[5], "Session": row[6]}
    conn.close()
    return classInfo

def getListStudentByID(classID):
    listStudents = []
    conn = connection()
    cursor = conn.cursor()
    sql = "call GetListStudentsToSendEmail(%s)"
    param = (classID, )
    cursor.execute(sql, param)
    for row in cursor.fetchall():
        listStudents.append({"mssv": row[0], "name": row[1], "email": row[2],
            "sessions": [ row[3], row[4], row[5], row[6],
            row[7], row[8], row[9], row[10],
            row[11], row[12], row[13], row[14],
            row[15], row[16], row[17]],
            'isSendEmail': row[18]})
    conn.close()

    return listStudents

def updateStatus(classID, mssv, status, isSendEmail):

    conn = connection()
    cursor = conn.cursor()

    sqlUpdate = "UPDATE Attended SET Status = %s, IsSendEmail = %s  Where ClassID = %s and StudentID = %s"
    param = (status, isSendEmail, classID, mssv,)
    cursor.execute(sqlUpdate, param)

    conn.commit()
    conn.close()

    return

def sendEmailWarning(classInfo, email):

    year = f"{classInfo['Year']-1}-{classInfo['Year']}"
    time = f"HK{classInfo['Semester']}/{year}"
    subject = f"Cảnh báo cấm thi {time}"
    body = f"""Thân chào sinh viên,

Sinh viên nhận được email này đang nằm trong diện bị cảnh báo cấm thi môn {classInfo['MaMH']} - {classInfo['TenMH']}, Nhóm {classInfo['Nhom']} {time}.
Nếu sinh viên nghỉ thêm bất kỳ buổi học nào thì sinh viên sẽ bị cấm thi.

Trân trọng,"""

    msg = Message(subject, sender = ('System Attend','SystemAttend@student.tdtu.edu.vn'), recipients = [email])
    msg.body = body
    mail.send(msg)

    return

def checkWarning(classID):

    # Lấy thông tin của lớp học và tính ra số buổi dc phép nghỉ
    classInfo = getClassInfoByID(classID)
    canAbsolute = classInfo['Session'] * 20 // 100
    #print(f"Số buổi được phép nghỉ: {canAbsolute}")
    # Lấy dữ liệu của tất cả sinh viên theo lớp
    listStudents = getListStudentByID(classID)
    # Kiểm tra từng sinh viên và update status cũng như gửi mail cảnh báo
    for student in listStudents:
        absolute = student['sessions'].count('V')
        #print(f"Số buổi vắng của sinh viên {student['mssv']}: {absolute}")
        if (canAbsolute == absolute):
            # Gửi Email cảnh báo và chuyển status sv thành warning
            if (student['isSendEmail'] != True):
                sendEmailWarning(classInfo, student['email'])
                updateStatus(classID, student['mssv'], 'Warning', 1)
        elif (canAbsolute < absolute):
            # Chuyyển status sinh viên thành Cấm thi
            updateStatus(classID, student['mssv'], 'Ban', 1)
    return

@app.route("/submitAttended", methods=["POST"])
def submitAttended():
    try:
        classID = request.json.get("classID", None)
        listStudents = request.json.get("listStudents", None)

        updateAttened(classID, listStudents)

        checkWarning(classID)

        result = {
            "success": True,
            "msg": "Submit Attended Successful!"
        }
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def CheckExistCourse(year, semester, createBy, data_group):
    conn = connection()
    cursor = conn.cursor()

    sql = 'SELECT COUNT(1) FROM Class WHERE Year = %s and Semester = %s and MaMH = %s and Nhom = %s and CreateBy = %s'
    for k,g in data_group:
        param = (year, semester, k[0], k[2],createBy,)
        cursor.execute(sql, param)
        if cursor.fetchone()[0]:
            conn.close()
            return True

    conn.close()
    return False

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

        #Group By maMH, Nhom
        data_group_check = groupby(data, lambda item: (item["Mã MH"], item["Tên MH"], item["Nhóm"]))
        data_group = groupby(data, lambda item: (item["Mã MH"], item["Tên MH"], item["Nhóm"]))
        # Check exist course:
        if (CheckExistCourse(year, semester, createBy, data_group_check)):
            result = {
                "success": False,
                "msg": "Please check your data for duplicate courses."
            }
        else:
            conn = connection()
            cursor = conn.cursor()
            for k,g in data_group:
                # k: Includes MaMH and Group to create 1 class
                # g: list student in class
                # Create Class
                sqlInsertClass = 'INSERT INTO Class (Year, Semester, MaMH, TenMH, Nhom, Sessions, CreateBy, CreateByName)' + \
                                'VALUES(%s,%s,%s,%s,%s,%s,%s,%s)'
                paramInsertClass = (year, semester, k[0], k[1], k[2], 15, createBy, createByName, )
                cursor.execute(sqlInsertClass, paramInsertClass)
                record_id = cursor.lastrowid
                conn.commit()
                # Create Student in Class
                for item in list(g):
                    sqlInsertStudentInClass = 'INSERT INTO Attended (ClassID, StudentID, Status, IsSendEmail) VALUES(%s,%s,%s,%s)'
                    paramInsertStudentInClass = (record_id, item.get('Mã số SV'), 'Active', False,)
                    cursor.execute(sqlInsertStudentInClass, paramInsertStudentInClass)
                    conn.commit()

            conn.close()

            result = {
                "success": True,
                "msg": "Import Course Success."
            }
    #except pyodbc.Error as e:
        #result = {"success": False, "msg": "Import Course Failed."}
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

def CheckExistStudents(data):
    conn = connection()
    cursor = conn.cursor()

    sql = 'SELECT COUNT(1) FROM Students WHERE ID = %s'
    for item in data:
        param = (item.get('Mã số SV'),)
        cursor.execute(sql, param)
        if cursor.fetchone()[0]:
            conn.close()
            return True

    conn.close()

    return False

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
        data_check = json.loads(data_json_string)
        data = json.loads(data_json_string)

        if (CheckExistStudents(data_check)):
            result = {
                "success": False,
                "msg": "Please check your data for duplicate student."
            }
        else:
            conn = connection()
            cursor = conn.cursor()

            for row in data:
                sqlInsertStudents = 'INSERT INTO Students (ID, Name, Sex, Email)' + \
                                'VALUES(%s,%s,%s,%s)'
                paramInsertStudents = (row.get('Mã số SV'), row.get('Họ lót') + ' ' + row.get('Tên'), row.get('Phái'), row.get('Email'), )
                cursor.execute(sqlInsertStudents, paramInsertStudents)

            conn.commit()
            conn.close()

            result = {
                "success": True,
                "msg": "Import Students Success."
            }
    #except pyodbc.Error as e:
        #result = {"success": False, "msg": "Import Students Failed."}
    except Exception as e:
        result = {"success": False, "msg": str(e)}

    return result

@app.route("/importFaces", methods=["POST"])
def importFaces():
    try:
        file = request.files.get('file')
        createBy = request.form.get('createBy')
        createByName = request.form.get('createByName')

        realPath = os.path.dirname(__file__)
        source_dir = os.path.join(realPath, 'OriginalFace')
        filename = file.filename
        # Lưu file về thư mục OriginalFace
        file.save(os.path.join(source_dir, filename))
        # Giải nén file
        zip_ref = zipfile.ZipFile(os.path.join(source_dir, filename), 'r')
        zip_ref.extractall(source_dir)
        zip_ref.close()
        # Xóa file
        os.remove(os.path.join(source_dir, filename))

        result = {
            "success": True,
            "msg": "Import Faces Success."
        }
    #except pyodbc.Error as e:
        #result = {"success": False, "msg": "Import Faces Failed."}
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
        sql = "Select MaMH, TenMH, Nhom, Sessions from Class where CreateBy = %s and Year = %s and Semester = %s"
        param = (userID, year, semester,)
        cursor.execute(sql, param)
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

def retrain():
    global modelSVC
    global modelFacenet
    global modelDetector

    # Thực hiện detect ảnh
    realPath = os.path.dirname(__file__)
    source_dir = os.path.join(realPath, 'OriginalFace')
    dest_dir = os.path.join(realPath, 'DetectFace')
    detect_face.detectData(modelDetector,source_dir,dest_dir)

    # Bắt đầu Embedding dữ liệu ảnh mới
    training.mainTraining(modelFacenet)

    # Fit lại model vừa mới train
    modelSVC = load_modelSVC()


@app.route("/retrainModel")
def retrainModel():

    retrain()

    result = {
        "success": True,
        "msg": "Train Data Success"
    }
    return result

def load_modelSVC():
    realPath = os.path.dirname(__file__)
    model = SVC(kernel='linear', probability=True)
    modelPath = os.path.join(realPath,'Model/faces-embeddings.npz')
    if os.path.exists(modelPath):
        data = load(os.path.join(realPath,'Model/faces-embeddings.npz'))

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
    realPath = os.path.dirname(__file__)
    mDectector = MTCNN(margin=10, select_largest=False)
    mSVC = load_modelSVC()
    mFacenet = load_model(os.path.join(realPath,'facenet_keras.h5'))#VGGFace(model='resnet50')#FaceNet()#
    # First using model

    image_test = cv2.imread(os.path.join(realPath,'image_loading.jpg'))

    msg, face_array = detect_face.dectect(mDectector, image_test)

    embedding = training.get_embedding(mFacenet, face_array)

    return mDectector, mFacenet, mSVC

@app.route("/reviewModel")
def reviewModel():
    realPath = os.path.dirname(__file__)
    result = 'thanh cong'
    global modelSVC
    data = load(os.path.join(realPath, 'Model/faces-embeddings.npz'))

    curtrainX, curtrainy, curtestX, curtesty = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (curtrainX.shape[0], curtestX.shape[0]))
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
    realPath = os.path.dirname(__file__)
    global modelSVC
    # load face embeddings
    data = load(os.path.join(realPath, 'Model/faces-embeddings.npz'))
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

    for selection in range(50):
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

@app.route("/testEmail")
def testEmail():
    msg = Message('This is Subject from email inside tdt', sender = ('System Attend','SystemAttend@student.tdtu.edu.vn'), recipients = ['51702014@student.tdtu.edu.vn'])
    msg.body = "This is new the email body"
    mail.send(msg)
    return 'Send Email success!'

@app.route("/testRemoveFace")
def testRemoveFace():
    realPath = os.path.dirname(__file__)

    data = load(os.path.join(realPath, 'Model/faces-embeddings.npz'))
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    listRemove = []
    listRemoveTest = []

    total_range = trainy.shape[0]
    for index in range(total_range):
        if trainy[index] == '51702014':
            listRemove.append(index)

    trainX = np.delete(trainX, listRemove, 0)
    trainy = np.delete(trainy, listRemove, 0)

    total_range_test = testy.shape[0]
    for index in range(total_range_test):
        if testy[index] == '51702014':
            listRemoveTest.append(index)

    testX = np.delete(testX, listRemoveTest, 0)
    testy = np.delete(testy, listRemoveTest, 0)

    savez_compressed(os.path.join(realPath, 'Model/faces-embeddings.npz'), trainX, trainy, testX, testy)

    #Remove folder detect
    dest_dir = os.path.join(realPath, 'DetectFace')
    student_dectect_Dir = os.path.join(dest_dir, '51702014')
    shutil.rmtree(student_dectect_Dir)

    origin_dir = os.path.join(realPath, 'OriginalFace')
    student_origin_Dir = os.path.join(origin_dir, '51702014')
    shutil.rmtree(student_origin_Dir)

    global modelSVC
    modelSVC = load_modelSVC()

    result = {"success": True, "msg": "Xóa dữ liệu khuôn mặt SV 51702014 thành công."}
    return result

@app.route("/manualload")
def manualload():
    global modelLabelEncoder
    global modelDetector
    global modelFacenet
    global modelSVC

    modelLabelEncoder = LabelEncoder()
    modelDetector, modelFacenet, modelSVC = load_AllModel()

    result = {"success": True, "msg": "done"}
    return result

@app.route("/testload")
def testload():
    realPath = os.path.dirname(__file__)
    global modelDetector

    image_test = cv2.imread(os.path.join(realPath,'image_loading.jpg'))
    msg, face_array = detect_face.dectect(modelDetector, image_test)

    result = {"success": True, "msg": "done"}
    return result

@app.route("/testdb")
def testdb():
    conn = connection()
    cursor = conn.cursor()
    qid = 1
    sql = "Select * from Attended where ID = %s"
    param = (qid,)
    cursor.execute(sql, param)
    conn.close()

    result = {"success": True, "msg": "done"}
    return result

modelLabelEncoder = None
modelDetector = None
modelFacenet = None
modelSVC = None

if __name__ == "__main__":
    manualload()
    app.run(debug=True)