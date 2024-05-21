
# Vera
### A Fashion Trends Prediction Project

This project utilizes a deep learning model to predict fashion trends by analyzing images from various fashion events. The goal is to identify the most frequently appearing fashion items (e.g., jackets, shoes, dresses) across a collection of images and determine what is trending.

## Overview

We developed a multi-label classification system that processes images, predicts multiple fashion item labels per image, and analyzes the frequency of these labels to determine fashion trends. This system uses TensorFlow and Keras for modeling and prediction, along with Scikit-learn's `MultiLabelBinarizer` for handling label transformations.

## Project Structure

- `load_and_preprocess_image.py`: Contains functions for image loading and preprocessing.
- `model_training.py`: Scripts for training the deep learning model.
- `predict.py`: Script for making predictions on new data.
- `utils.py`: Helper functions for various tasks like downloading images and converting label indices to human-readable labels.
- `data/`: Directory containing training and validation datasets.
- `models/`: Stored pre-trained models.
- `outputs/`: Predicted labels and analysis results.

## Model Description

The model is a convolutional neural network (CNN) designed for multi-label image classification. It's structured as follows:

1. **Input Layer**: Accepts images resized to 128x128 pixels.
2. **Convolutional and Pooling Layers**: Multiple layers to extract features from images.
3. **Dense Layers**: Fully connected layers to interpret features, including dropout for regularization.
4. **Output Layer**: Uses sigmoid activations to independently predict the presence of each label.

The model predicts probabilities for each label, and a threshold of 0.3 is used to decide label assignments.

## Key Functions

### Image Preprocessing

Images are loaded, decoded, resized to 128x128 pixels, and normalized to have pixel values between 0 and 1.

```python
def load_and_preprocess_image_for_prediction(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img /= 255.0
    return img
```

### Prediction

Predictions are made by processing each image through the model, extracting predicted label indices based on a set threshold, and converting these indices to human-readable labels.

```python
def predict_labels_for_images(download_dir, model):
    predicted_labels = {}
    for filename in os.listdir(download_dir):
        if filename.endswith(".jpg"):
            file_path = os.path.join(download_dir, filename)
            image = load_and_preprocess_image_for_prediction(file_path)
            image = tf.expand_dims(image, 0)
            predictions = model.predict(image)
            label_indices = np.where(predictions[0] > 0.3)[0]
            labels = [mlb.classes_[i] for i in label_indices]
            predicted_labels[filename] = labels
    return predicted_labels
```

### Trend Analysis

After prediction, label frequencies are analyzed to determine which items are most prevalent.

```python
def find_trending_label(all_labels):
    from collections import Counter
    label_counts = Counter(all_labels)
    trending_label = label_counts.most_common(1)[0]
    return trending_label, label_counts
```

## Usage

To run the prediction and analyze trends, execute the `predict.py` script after placing images in the specified directory. The script outputs the predicted labels for each image and the overall trending fashion items.

## Requirements

- TensorFlow 2.x
- Scikit-Learn
- NumPy
- Pillow

Install the necessary packages using:

```bash
pip install tensorflow numpy scikit-learn pillow
```

## Conclusion

This project demonstrates the capability of deep learning to analyze and predict fashion trends from images. By leveraging CNNs, we can not only identify multiple items in a single image but also analyze the prevalence of these items to gauge what's trending in the fashion world.

---

This README provides an overview of the project components, model structure, key functionalities, and how to use the system. Adjust the paths and other specific details according to your actual project setup.
<!-- #endregion -->

```python

```
