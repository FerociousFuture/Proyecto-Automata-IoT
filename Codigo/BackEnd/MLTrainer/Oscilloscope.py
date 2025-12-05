# Proyecto-Automata-IoT/Codigo/BackEnd/MLTrainer/Oscilloscope.py

class Input():
    def __init__(self, accX, accY, accZ, angRX, angRY, angRZ):
        self.accX = accX
        self.accY = accY
        self.accZ = accZ
        self.angRX = angRX
        self.angRY = angRY
        self.angRZ = angRZ
    
    def toDictionary(self):
        # Devuelve un diccionario simple (sin anidar en un conjunto)
        return {"accX": self.accX, "accY": self.accY, "accZ": self.accZ,
                "angRX": self.angRX, "angRY": self.angRY,"angRZ": self.angRZ}

class Mpu6050():
    def __init__(self, data = []):
        self.data = data

    def insertData(self, input):
        self.data.append(input)

    def getData(self):
        return self.data

    def getDataDictList(self):
        # Mapea los objetos Input a una lista de diccionarios
        dictionary = [input.toDictionary() for input in self.data]
        return dictionary