import os
import cv2
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
import numpy as np

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
def training_Array(Images,dataDir):
    X, y = list(), list()
    for img in Images:
        currentDir = os.path.join(dataDir, img.name)
        faces = []
        for path in img.image_paths:
            currentImg = cv2.imread(os.path.join(currentDir,path))
            faces.append(currentImg) 

        labels = [img.name for _ in range(len(faces))]

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

def split_data(dataDir, currentStudentTrain = []):

    train_set = []
    test_set = []

    for dir_person in os.listdir(dataDir):
        # Nếu đã tồn tại trong dữ liệu train thì không cần train nữa
        if (currentStudentTrain != [] and dir_person in currentStudentTrain):
            continue
        
        listImages = os.listdir(os.path.join(dataDir,dir_person))

        dataTest = listImages[-5:]    
        dataTrain = set(listImages) - set(dataTest)
    
        train_set.append(ImageClass(dir_person, dataTrain))
        test_set.append(ImageClass(dir_person, dataTest))

    return train_set, test_set

def mainTraining(modelFacenet):
    realPath = os.path.dirname(__file__)
    detect_dir = os.path.join(realPath, 'DetectFace')
    # Nếu đã từng Embedding dữ liệu rồi
    if os.path.exists(os.path.join(realPath,'Model/faces-dataset.npz')):
        # Load Model cũ lên
        data = load('Model/faces-dataset.npz')
        currTrainX, currTrainy, currTestX, currTesty = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        currentStudentTrain = list(set(currTrainy))

        trainImages, testImages = split_data(detect_dir, currentStudentTrain)

        if (trainImages != [] and testImages != []):
        
            trainX, trainY = training_Array(trainImages, detect_dir)
            testX, testY = training_Array(testImages, detect_dir)

            newTrainX = []
            for face_pixels in trainX:
                embedding = get_embedding(modelFacenet, face_pixels)
                newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)

            newTestX = []
            for face_pixels in testX:
                embedding = get_embedding(modelFacenet, face_pixels)
                newTestX.append(embedding)
            newTestX = asarray(newTestX)

            resultTrainX = np.append(currTrainX, newTrainX , axis=0)
            resultTrainy = np.append(currTrainy, trainY , axis=0)
            resultTestX = np.append(currTestX, newTestX , axis=0)
            resultTesty = np.append(currTesty, testY , axis=0)

            savez_compressed('Model/faces-dataset.npz', resultTrainX, resultTrainy, resultTestX, resultTesty)
    # Dữ liệu mới
    else:
        trainImages, testImages = split_data(detect_dir)

        trainX, trainY = training_Array(trainImages,detect_dir)
        testX, testY = training_Array(testImages,detect_dir)

        newTrainX = []
        for face_pixels in trainX:
            embedding = get_embedding(modelFacenet, face_pixels)
            newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)

        newTestX = []
        for face_pixels in testX:
            embedding = get_embedding(modelFacenet, face_pixels)
            newTestX.append(embedding)
        newTestX = asarray(newTestX)

        savez_compressed('Model/faces-dataset.npz', newTrainX, trainY, newTestX, testY)

    return