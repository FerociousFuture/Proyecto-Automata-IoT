# Proyecto-Automata-IoT/Codigo/BackEnd/MLTrainer/Trainer.py

import numpy as np
from .Oscilloscope import Input, Mpu6050

def transform_to_numpy_array(input_objects):
    """Convierte una lista de objetos Input en un array NumPy (6 ejes)."""
    data_list = []
    for obj in input_objects:
        data_list.append([obj.accX, obj.accY, obj.accZ, obj.angRX, obj.angRY, obj.angRZ])
        
    return np.array(data_list)

# Esta es la función que usarás para empezar el entrenamiento.
def toArray(data_objects):
    """La función que usarás para transformar la data cargada a un array NumPy."""
    return transform_to_numpy_array(data_objects)

# Aquí puedes añadir las funciones para entrenar, predecir, etc.
# def train_model(numpy_data, labels):
#     # Lógica de Machine Learning con TensorFlow/scikit-learn
#     pass