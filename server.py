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

def loadListStudent_Code(code):
    students = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("SELECT s.MSSV, s.Name, m.Present, m.Absence, m.Status FROM Mapping m join Class c on c.ID = m.ClassID join Students s on s.ID = m.StudentID WHERE c.Code = ?",code)
    for row in cursor.fetchall():
        students.append({"mssv": row[0], "name": row[1], "present": row[2], "absence": row[3], "status": row[4]})
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