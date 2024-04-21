# Disease Classification using Image Features

This project aims to classify diseases in images using a combination of color histogram, texture features, and pretrained InceptionV3 features. It utilizes a Support Vector Machine (SVM) model trained on extracted features to classify diseases accurately.

## Dependencies:

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- scikit-image (`skimage`)
- scikit-learn (`scikit-learn`)
- TensorFlow (`tensorflow`)
- tqdm (`tqdm`)

You can install the required dependencies via pip:

```
pip install opencv-python numpy scikit-image scikit-learn tensorflow tqdm
```

## Usage:

1. Ensure that you have a dataset with images of diseases and their corresponding labels.
2. Train an SVM model using the provided scripts or a similar approach.
3. Save the trained SVM model weights to a file (e.g., `svm_model_weights.h5`).
4. Save the test features (`X_test.npy`) and test labels (`y_test.npy`) to evaluate the model's accuracy.
5. Run the Python script provided in this repository.
6. Enter the path of the image you want to classify when prompted.
7. The predicted disease class will be displayed along with the classification accuracy.

## Code Structure:

- **Image Preprocessing (`preprocess_image(image_path)`)**: This function preprocesses the input image by resizing it to the input size expected by InceptionV3 and applying appropriate transformations.

- **Feature Extraction (`extract_color_histogram(image)`, `extract_texture_features(image)`, `extract_inception_features(image_path)`)**: These functions extract color histogram, texture features using Grey Level Co-occurrence Matrix (GLCM), and pretrained InceptionV3 features from the input image, respectively.

- **Load SVM Model (`load_svm_model(model_path)`)**: This function loads the trained SVM model from the specified file (`svm_model_weights.h5`).

- **Classify Disease (`classify_disease(image_path, svm_model)`)**: This function classifies the disease in the input image by extracting features and predicting the disease class using the loaded SVM model.

- **Main Function**: The main function loads the trained SVM model, evaluates its accuracy on the test set, and then enters a loop to classify diseases in user-provided images.

## Notes:

- Ensure that you have a suitable dataset with images of diseases and their corresponding labels for training and evaluation.
- Train the SVM model using appropriate features extracted from the dataset. You may need to fine-tune the model hyperparameters for optimal performance.
- Save the trained SVM model weights and test set features/labels to files for future evaluation.
- Experiment with different image preprocessing techniques and feature extraction methods to improve classification accuracy.
- This project provides a basic framework for disease classification using image features. You can extend it by incorporating more sophisticated feature extraction methods or deep learning models.

## Contributors:

- Prashanth-Devarahatti && ChatGPT



---
Feel free to customize the README further with additional sections or details as needed!
