from PIL import Image
import cv2
import os
from numpy import asarray

def detectData(model, source_dir, dest_dir, required_size = (160,160)):
    # Nếu chưa có folder DetectFace thì tạo mới
    if os.path.exists(dest_dir)==False:
        os.mkdir(dest_dir)
    # Chạy từng Folder của mỗi người trong thư mục Original Face
    for dir_person in os.listdir(source_dir):
        # dir_person là tên thư mục với MSSV
        source_path = os.path.join(source_dir,dir_person)
        # Kiểm tra nếu ko phải folder thì bỏ qua
        if not os.path.isdir(source_path):
            continue

        # Tạo ra folder của người đó bên phía đã detect nếu chưa có
        detect_path = os.path.join(dest_dir,dir_person)
        if os.path.exists(detect_path)==False:
            os.mkdir(detect_path)
        # Load tất cả file hình trong thư mục của người đó bên phía Original
        source_list = os.listdir(source_path)
        # Bắt đầu detect khuôn mặt trong những bức ảnh đó
        for f in source_list:
            # f_path là file hình gốc; dest_path là file kết quả
            f_path=os.path.join(source_path, f)
            dest_path=os.path.join(detect_path,f)
            # Nếu hình đó đã detect rồi thì ko cần thực hiện lại lần nữa
            if os.path.exists(dest_path):
                continue
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
                # Chuyển về màu gốc
                image = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                # Resize khuôn mặt về 1 kích cỡ
                resultImage = Image.fromarray(image)
                resultImage = resultImage.resize(required_size)
                face_array = asarray(resultImage)
                # Sau đó lưu vào folde kết quả
                cv2.imwrite(dest_path, face_array)

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