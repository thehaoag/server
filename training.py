import os
import cv2
import random
from keras_facenet import FaceNet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from PIL import Image

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  

def facenet(dataDir):
    trainImages, testImages = split_data(dataDir)

    trainX, trainY = training_Array(trainImages,dataDir)
    testX, testY = training_Array(testImages,dataDir)
    
    model = FaceNet()

    newTrainX = []
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)

    newTestX = []
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)

    savez_compressed('Model/faces-dataset.npz', newTrainX, trainY, newTestX, testY)

    return

def training_Array(Images,dataDir):
    required_size=(160, 160)
    X, y = list(), list()
    for img in Images:
        currentDir = os.path.join(dataDir, img.name)
        faces = []
        for path in img.image_paths:
            currentImg = cv2.imread(os.path.join(currentDir,path))
            image = Image.fromarray(currentImg)
            image = image.resize(required_size)
            face = asarray(image)
            faces.append(face) 
        labels = [img.name for _ in range(len(faces))]

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    dectections = model.extract(face_pixels, threshold=0.8)
    #mean, std = face_pixels.mean(), face_pixels.std()
    #face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

def split_data(dataDir):

    train_set = []
    test_set = []

    for dir_person in os.listdir(dataDir):
        listImages = os.listdir(os.path.join(dataDir,dir_person))

        dataTest = random.sample(listImages, k=5)        
        dataTrain = set(listImages) - set(dataTest)
    
        train_set.append(ImageClass(dir_person, dataTrain))
        test_set.append(ImageClass(dir_person, dataTest))

    return train_set, test_set