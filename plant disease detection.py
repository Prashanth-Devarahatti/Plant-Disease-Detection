import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm

# Step 1: Image preprocessing

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))  # InceptionV3 input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = preprocess_input(img)
    return img

# Step 2: Feature extraction

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = greycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    return [contrast, energy, homogeneity, correlation]

# Step 3: Load pretrained InceptionV3 model for feature extraction

def extract_inception_features(image_path):
    model = InceptionV3(weights='imagenet', include_top=False)
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

# Step 4: Load trained SVM model

def load_svm_model(model_path):
    svm_model = SVC(kernel='rbf', C=1000, gamma='scale')
    svm_model.load_weights(model_path)
    return svm_model

# Step 5: Classify disease

def classify_disease(image_path, svm_model):
    # Extract features
    img = preprocess_image(image_path)
    color_hist = extract_color_histogram(img)
    texture_features = extract_texture_features(img)
    inception_features = extract_inception_features(image_path)
    
    # Concatenate features
    features = np.concatenate([color_hist, texture_features, inception_features])
    
    # Reshape features
    features = features.reshape(1, -1)
    
    # Predict disease
    disease_class = svm_model.predict(features)
    
    # Return predicted disease class and accuracy
    return disease_class

if __name__ == "__main__":
    # Load trained SVM model
    model_path = 'svm_model_weights.h5'  # Path to trained SVM model weights
    svm_model = load_svm_model(model_path)

    # Load dataset to calculate accuracy
    X_test = np.load('X_test.npy')  # Load test features
    y_test = np.load('y_test.npy')  # Load test labels

    # Evaluate SVM model on test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", accuracy)

    while True:
        # Input image path
        image_path = input("Enter the path of the image: ")

        # Classify disease
        try:
            disease_class = classify_disease(image_path, svm_model)
            print("Predicted disease class:", disease_class)
            print("Accuracy:", accuracy)
        except:
            print("Error: Invalid image path or unable to classify disease. Please try again.")
