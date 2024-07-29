import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare the dataset
breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Build and compile the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train_std.shape[1],)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train_std, Y_train, epochs=10, validation_data=(X_test_std, Y_test))

# Save the model
model.save('D:\save.h5')
