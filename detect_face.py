from PIL import Image
import cv2
import os
from numpy import asarray

def detectData(model, Images, required_size = (160,160)):
    realPath = os.path.dirname(__file__)
    source_path = os.path.join(realPath, 'OriginalFace')
    X, y = list(), list()
    # Chạy từng Folder của mỗi người trong thư mục Original Face
    for person in Images:
        faces = list()
        # Bắt đầu detect khuôn mặt trong những bức ảnh đó
        for f in person.image_paths:
            # f_path là file hình gốc; dest_path là file kết quả
            dir_person = os.path.join(source_path,person.name)
            f_path=os.path.join(dir_person, f)
            # Nếu chưa thì chuyển hình đó thành hình với 3 màu RGB
            img = cv2.cvtColor(cv2.imread(f_path), cv2.COLOR_BGR2RGB)
            # Thực hiện detect khuôn mặt trên ảnh
            bounding_box, _ = model.detect(img)
            # Nếu đã tìm thấy khuôn mặt thì mới thực hiện
            if bounding_box is not None:
                # Nếu bức ảnh có nhiều hơn 1 khuôn mặt thì lấy khuôn mặt đầu tiên
                # Sau khi tìm thấy 1 khuôn mặt thì ta cắt khuôn mặt từ ảnh gốc
                box = bounding_box[0]
                # Nếu vượt quá kích thước ảnh thì trả về 0
                if (int(box[0]) < 0 ):
                    box[0] = 0
                if (int(box[1]) < 0 ):
                    box[1] = 0
                face = img[int(box[1]) : int(box[3]),int(box[0]) : int(box[2])]
                # Resize khuôn mặt về 1 kích cỡ
                resultImage = Image.fromarray(face)
                resultImage = resultImage.resize(required_size)
                face_array = asarray(resultImage)
                faces.append(face_array)

        labels = [person.name for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def dectect(model, image, required_size = (160,160)):
    face_array = []
    errorMsg = ''
    # Chuyển hình thành màu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Thực hiện detect khuôn mặt trên ảnh
    bounding_box, _ = model.detect(image_rgb)
    # Nếu đã tìm thấy khuôn mặt thì mới thực hiện
    if bounding_box is not None:
        # Nếu bức ảnh có nhiều hơn 1 khuôn mặt thì sẽ bỏ qua
        faces_found = bounding_box.shape[0]
        if faces_found == 1:
            # Sau khi tìm thấy 1 khuôn mặt thì ta cắt khuôn mặt từ ảnh gốc
            box = bounding_box[0]
            # Nếu vượt quá kích thước ảnh thì trả về 0
            if (int(box[0]) < 0 ):
                box[0] = 0
            if (int(box[1]) < 0 ):
                box[1] = 0
            face = image[int(box[1]): int(box[3]),int(box[0]): int(box[2])]
            # Chuyển khuôn mặt thành mãng pixel và resize
            newFace = Image.fromarray(face)
            newImg = newFace.resize(required_size)
            face_array = asarray(newImg)
        else:
            errorMsg = 'Có nhiều hơn 1 khuôn mặt trên camera.'
    else:
        errorMsg = 'Không tìm thấy khuôn mặt trên camera.'
    return errorMsg, face_array