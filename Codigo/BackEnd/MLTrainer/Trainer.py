# Proyecto-Automata-IoT/Codigo/BackEnd/MLTrainer/Trainer.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Nota: Las funciones transform_to_numpy_array y toArray ya no son necesarias aquí 
# porque data_processor.py maneja esa lógica antes de llamar al entrenador.

def create_lstm_model(input_shape, num_classes):
    """Crea y compila un modelo básico de red neuronal LSTM para clasificación."""
    
    model = Sequential([
        # LSTM es excelente para series de tiempo
        LSTM(64, input_shape=input_shape, return_sequences=True), 
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax') # Softmax para clasificación multi-clase
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Modelo LSTM creado y compilado.")
    model.summary()
    return model

def train_gesture_recognizer(X, y, encoder, epochs=20, batch_size=32):
    """
    Divide los datos, entrena el modelo y guarda los resultados.
    """
    if X is None or y is None:
        print("No hay datos para entrenar.")
        return

    # Dividir los datos en conjuntos de entrenamiento y prueba (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    num_classes = len(encoder.classes_)
    input_shape = (X_train.shape[1], X_train.shape[2]) # (SEQUENCE_LENGTH, 6)
    
    model = create_lstm_model(input_shape, num_classes)
    
    # Entrenamiento
    print("\n--- INICIANDO ENTRENAMIENTO ---")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluación final
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión final del modelo: {accuracy:.4f}")
    
    # Guardar el modelo para usarlo en el autómata (Sprint 4)
    model_save_path = "gesture_model.h5"
    model.save(model_save_path)
    print(f"Modelo guardado en: {model_save_path}")


if __name__ == '__main__':
    # Importar el preparador de datos para iniciar el flujo de entrenamiento
    from data_processor import prepare_data_for_training
    
    X, y, encoder = prepare_data_for_training()
    
    if X is not None:
        # Iniciar el entrenamiento con los datos preparados
        train_gesture_recognizer(X, y, encoder)