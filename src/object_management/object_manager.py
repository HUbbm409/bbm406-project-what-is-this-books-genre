import pickle


class ObjectManager:
    def __init__(self):
        pass
    # record the object
    def RecordObject(self,myObject, modelName):
        fileName = modelName
        fileObject = open(fileName, 'wb')
        pickle.dump(myObject, fileObject)
        fileObject.close()


    # read the object
    def ReadObject(self,modelName):
        fileObject = open(modelName, 'rb')
        b = pickle.load(fileObject)
        return b
