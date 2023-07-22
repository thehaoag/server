import os
import cv2
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
import numpy as np
import detect_face
from keras_vggface.utils import preprocess_input

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

def get_embedding(model, face_pixels):
    #Facenet H5
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

    #Facenet
    #samples = expand_dims(face_pixels, axis=0)
    #yhat = model.embeddings(samples)
    #return yhat[0]

    #VGGFace
    #face_pixels = face_pixels.astype('float32')
    #samples = expand_dims(face_pixels, axis=0)
    #samples = preprocess_input(samples, version=2)
    #yhat = model.predict(samples)
    #return yhat[0]

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

def mainTraining(modelDetector, modelFacenet):
    realPath = os.path.dirname(__file__)
    source_dir = os.path.join(realPath, 'OriginalFace')
    # Nếu đã từng Embedding dữ liệu rồi
    if os.path.exists(os.path.join(realPath,'Model/faces-dataset.npz')):
        # Load Model cũ lên
        data = load('Model/faces-dataset.npz')
        currTrainX, currTrainy, currTestX, currTesty = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        currentStudentTrain = list(set(currTrainy))

        trainImages, testImages = split_data(source_dir, currentStudentTrain)

        if (trainImages != [] and testImages != []):
            newTrainX, newTrainy = detect_face.detectData(modelDetector, trainImages)
            newTestX, newTesty = detect_face.detectData(modelDetector, testImages)

            resultTrainX = np.append(currTrainX, newTrainX , axis=0)
            resultTrainy = np.append(currTrainy, newTrainy , axis=0)
            resultTestX = np.append(currTestX, newTestX , axis=0)
            resultTesty = np.append(currTesty, newTesty , axis=0)

            savez_compressed('Model/faces-dataset.npz', resultTrainX, resultTrainy, resultTestX, resultTesty)

            newEmbeddingTrainX = []
            for face_pixels in newTrainX:
                embedding = get_embedding(modelFacenet, face_pixels)
                newEmbeddingTrainX.append(embedding)
            newEmbeddingTrainX = asarray(newEmbeddingTrainX)

            newEmbeddingTestX = []
            for face_pixels in newTestX:
                embedding = get_embedding(modelFacenet, face_pixels)
                newEmbeddingTestX.append(embedding)
            newEmbeddingTestX = asarray(newEmbeddingTestX)
            
            embbedingData = load('Model/faces-embeddings.npz')
            currTrainX, currTrainy, currTestX, currTesty = embbedingData['arr_0'], embbedingData['arr_1'], embbedingData['arr_2'], embbedingData['arr_3']

            resultTrainX = np.append(currTrainX, newEmbeddingTrainX , axis=0)
            resultTrainy = np.append(currTrainy, newTrainy , axis=0)
            resultTestX = np.append(currTestX, newEmbeddingTestX , axis=0)
            resultTesty = np.append(currTesty, newTesty , axis=0)
            
            savez_compressed('Model/faces-embeddings.npz', resultTrainX, resultTrainy, resultTestX, resultTesty)
    # Dữ liệu mới
    else:

        trainImages, testImages = split_data(source_dir)

        trainX, trainy = detect_face.detectData(modelDetector, trainImages)
        testX, testy = detect_face.detectData(modelDetector, testImages)

        savez_compressed('Model/faces-dataset.npz', trainX, trainy, testX, testy)

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

        savez_compressed('Model/faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

