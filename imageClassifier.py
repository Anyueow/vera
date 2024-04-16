import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalMaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd

styles_df = pd.read_csv('styles.csv')

#import dask.dataframe as dd

IMG_PATH = '/Users/nataliehammel/Downloads/fashion-dataset/images/'


def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional


def img_path(img):
    return IMG_PATH + img


def load_image(img):
    return cv2.imread(img_path(img))


# generation of a dictionary of (title, images)
figures = {'im' + str(i): load_image(row.image, True) for i, row in styles_df.sample(1).iterrows()}
plot_figures(figures, 2, 3)

"""""
# Input Shape
img_width, img_height, _ = load_image(styles_df.iloc[0].image).shape

# Pre-Trained Model
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape = (img_width, img_height, 3))
base_model.trainable = False

# Add Layer Embedding
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def get_embedding(model, img_name):

    # Reshape
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    # img to Array
    x = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # Pre process Input
    x = preprocess_input(x)
    return model.predict(x, verbose=0).reshape(-1)

emb = get_embedding(model, styles_df.iloc[59].image)
print(emb.shape)

plt.imshow(cv2.cvtColor(load_image(styles_df.iloc[59].image), cv2.COLOR_BGR2RGB))
print(emb)
"""
