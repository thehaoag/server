from flask import Flask, render_template, Response, request, jsonify
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
import time
import os
from datetime import datetime, timedelta, timezone
from flask_jwt_extended import create_access_token, get_jwt, get_jwt_identity, unset_jwt_cookies, jwt_required, JWTManager

app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = "SECRET_KEY_PROJECT_ATTENDED_SYSTEM"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

@app.route('/login', methods=["POST"])
def login():
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

    return response 
    

@app.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
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

def loadListStudent(year,semester,maMH,group):
    students = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("exec GetListStudents ?,?,?,?", year, semester, maMH, group)
    for row in cursor.fetchall():
        students.append({"mssv": row[0], "name": row[1], "session1": row[2], "session2": row[3], "session3": row[4],"session4": row[4],
        "session5": row[2], "session6": row[3], "session7": row[4],"session8": row[2], "session9": row[3], "session10": row[4],
        "session11": row[4], "session12": row[4], "session13": row[4], "session14": row[4], "session15": row[4]})
    conn.close()

    return students

def loadListStudent_Attend(year,semester,maMH,group,session):
    students = []
    classID = 0
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("exec GetListStudents ?,?,?,?", year, semester, maMH, group)
    for row in cursor.fetchall():
        classID = row[0]
        date = row[session+2]
        if (date):
            date = row[session+2].strftime("%d/%m/%Y")
        students.append({"mssv": row[1], "name": row[2], "datesession": date, "session": session})
    conn.close()

    return students, classID

@app.route("/getListStudents_Attend", methods=["POST"])
def getListStudents_Attend():
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

    return result

@app.route("/getListStudents", methods=["POST"])
def getListStudents():
    year = request.json.get("year", None)
    semester = request.json.get("semester", None)
    maMH = request.json.get("maMH", None)
    group = request.json.get("group", None)

    students = loadListStudent(year,semester,maMH,group)

    if students != [] and len(students) > 0:
        result = {
            "success": True,
            "currentCode": code,
            "listStudents": students
        }
    else:
        result = {
            "success": False,
            "msg": "Không tìm thấy mã lớp " + code
        }

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

    start1 = time.time()
    students = loadListStudent_Code(code)
    print("---Load List Student: %s seconds ---" % (time.time() - start1))

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

        #Predict
        start7 = time.time()
        # Sử dụng model SVC để dự đoán
        samples = expand_dims(embedding, axis=0)
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

    return result

def updateAttened(classID, listStudents):
    conn = connection()
    cursor = conn.cursor()
    for student in listStudents:
        sqlUpdate = "UPDATE Attended SET Buoi" + str(student.get('session')) +" = ? Where ClassID = ? and StudentID = ?"
        cursor.execute(sqlUpdate, student.get('datesession'), classID, student.get('mssv'))
    
    conn.commit()
    conn.close()
    return

@app.route("/submitAttended", methods=["POST"])
def submitAttended():
    classID = request.json.get("classID", None)
    listStudents = request.json.get("listStudents", None)

    updateAttened(classID, listStudents)

    result = {
        "success": True,
        "msg": "Submit Attended Successful!"
    }

    return result

def retrainModel():
    ErrorMsg = ''
    # Thực hiện detect ảnh
    realPath = os.path.dirname(__file__)
    source_dir = os.path.join(realPath, 'OriginalFace')
    dest_dir = os.path.join(realPath, 'DetectFace') 
    detect_face.detectData(modelDetector,source_dir,dest_dir)
    # Bắt đầu Embedding dữ liệu ảnh mới
    global modelFacenet
    training.mainTraining(modelFacenet)
    # Fit lại model vừa mới train
    global modelSVC
    modelSVC = load_modelSVC()

    return ErrorMsg

@app.route("/themsinhvien")
def themsinhvien():
    result = 'thanh cong'
    ErrorMsg = retrainModel()
    return result

def load_modelSVC():
    model = SVC(kernel='linear', probability=True)
    modelPath = os.path.join(os.path.dirname(__file__),'Model/faces-dataset.npz')
    if os.path.exists(modelPath):
        data = load('Model/faces-dataset.npz')

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

def load_modelFaceNet():
    model = FaceNet()
    realPath = os.path.dirname(__file__)
    img = cv2.imread(os.path.join(realPath,"camera.jpg"))
    face_pixels = asarray(img)
    face_pixels = face_pixels.astype('float32')
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return model

def load_AllModel():
    mDectector = MTCNN(margin=20, select_largest=False)
    mSVC = load_modelSVC()
    mFacenet = load_modelFaceNet()
    return mDectector, mFacenet, mSVC

if __name__ == "__main__":
    currentCamera = []
    modelLabelEncoder = LabelEncoder()
    modelDetector, modelFacenet, modelSVC = load_AllModel()
    app.run(debug=True)