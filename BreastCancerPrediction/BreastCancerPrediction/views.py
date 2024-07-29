from django.shortcuts import render
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from tensorflow import keras

# Initialize global variables
model = None
scaler = None

def initialize_model_and_scaler():
    global model, scaler

    # Load dataset
    breast_cancer_dataset = load_breast_cancer()
    feature_names = breast_cancer_dataset.feature_names
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=feature_names)
    data_frame['label'] = breast_cancer_dataset.target

    # Select 20 features (for simplicity, using first 20 features here; adjust as needed)
    selected_features = feature_names[:20]
    X = data_frame[selected_features]
    y = data_frame['label']

    # Initialize and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Define and train the model
    tf.random.set_seed(3)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X.shape[1],)),  # 20 features
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model.fit(X_train_std, Y_train, epochs=10, validation_data=(X_test_std, Y_test))

# Initialize model and scaler
initialize_model_and_scaler()

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        # Extract features from the request
        features = np.array([
            float(request.POST.get('mean_radius', 0)),
            float(request.POST.get('mean_texture', 0)),
            float(request.POST.get('mean_perimeter', 0)),
            float(request.POST.get('mean_area', 0)),
            float(request.POST.get('mean_smoothness', 0)),
            float(request.POST.get('mean_compactness', 0)),
            float(request.POST.get('mean_concavity', 0)),
            float(request.POST.get('mean_concave_points', 0)),
            float(request.POST.get('mean_symmetry', 0)),
            float(request.POST.get('mean_fractal_dimension', 0)),
            float(request.POST.get('worst_radius', 0)),
            float(request.POST.get('worst_texture', 0)),
            float(request.POST.get('worst_perimeter', 0)),
            float(request.POST.get('worst_area', 0)),
            float(request.POST.get('worst_smoothness', 0)),
            float(request.POST.get('worst_compactness', 0)),
            float(request.POST.get('worst_concavity', 0)),
            float(request.POST.get('worst_concave_points', 0)),
            float(request.POST.get('worst_symmetry', 0)),
            float(request.POST.get('worst_fractal_dimension', 0))
        ])

        # Ensure the features array has the correct shape (1, 20)
        features = features.reshape(1, -1)

        # Standardize the input features
        features_std = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features_std)
        result = np.argmax(prediction, axis=1)[0]

        # Interpret the result
        result_text = "Benign" if result == 1 else "Malignant"

        return render(request, 'predict.html', {'result': result_text})

    return render(request, 'predict.html')
