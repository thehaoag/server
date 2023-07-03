from mtcnn import MTCNN
import cv2
import os

def detect_face(source_dir, dest_dir):
    # Khai báo thư viên mtcnn
    detector = MTCNN()
    # Nếu chưa có folder thì tạo mới
    if os.path.exists(dest_dir)==False:
        os.mkdir(dest_dir)

    for dir_person in os.listdir(source_dir):
        # Kiểm tra nếu ko phải folder thì bỏ qua
        source_path = os.path.join(source_dir,dir_person)
        if not os.path.isdir(source_path):
            continue

        # Tạo ra folder của người đó nếu chưa có
        detect_path = os.path.join(dest_dir,dir_person)
        if os.path.exists(detect_path)==False:
            os.mkdir(detect_path)

        source_list = os.listdir(source_path)
        for f in source_list:
            f_path=os.path.join(source_path, f)
            dest_path=os.path.join(detect_path,f)
            #Nếu đã detect rồi thì ko cần thực hiện lại lần nữa
            if os.path.exists(dest_path):
                continue
            img = cv2.cvtColor(cv2.imread(f_path), cv2.COLOR_BGR2RGB)
            data=detector.detect_faces(img)
            if data !=[]:
                first = data[0]

                box=first['box']  # get the box for each face                

                img=img[box[1]: box[1]+box[3],box[0]: box[0]+ box[2]]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dest_path, img)

    return
