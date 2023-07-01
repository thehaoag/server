from flask import Flask,render_template,Response
import cv2
import pyodbc

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

@app.route("/loadcamera")
def loadcamera():
    return render_template('./camera.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        #read camera frame
        
        success,frame = camera.read()
        if not success:
            break
        
        cv2.imwrite('demo.jpg', frame)

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/getListStudents/<string:code>")
def getListStudents(code):
    
    students = []
    conn = connection()
    cursor = conn.cursor()
    cursor.execute("SELECT s.MSSV, s.Name FROM Mapping m join Class c on c.ID = m.ClassID join Students s on s.ID = m.StudentID WHERE c.Code = ?",code)
    for row in cursor.fetchall():
        students.append({"mssv": row[0], "name": row[1]})
    conn.close()

    count = len(students)
    #paging = [students[i:i+per_page] for i in range(0, len(students), per_page)]

    result = {
        "total": count,
        "listStudents": students
    }

    return result

if __name__ == "__main__":
    app.run(debug=True)