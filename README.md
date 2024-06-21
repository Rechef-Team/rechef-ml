# Rechef-ML Model
<img alt="banner ternaku" src="https://github.com/Rechef-Team/.github/blob/main/profile/banner.png?raw=true"><br>
## Introduction
Rechef is an innovative mobile application designed to transform the way you cook. By recognizing the ingredients you have in your kitchen, Rechef recommends delicious and tailored recipes that make the most out of what you already have.

## Description
This repository contains Machine Learning models that have been trained to perform the task of classifying Food ingredients.



### Dataset Used

[Food Ingredients Classification](https://github.com/Rechef-Team/rechef-ml/blob/main/DatasetBahanV1.zip) 8 classes: tomato, tofu, tempeh, chicken meat, chili, shallots, garlic, and eggs.

### Prerequisites

Before using this model, make sure you have the following software installed:

1. Python (version 3.0 or later)
2. The Python packages listed in the requirements.txt file.

### Installation

1. **Clone this repository to your local device:**

    ```bash
    git clone https://github.com/Rechef-Team/rechef-ml.git
    ```

2. **Go to the repository directory:**

    ```bash
    cd rechef-ml
    ```

3. **(Optional) Create and activate the virtual environment:**

    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Google Colab

You can run this project in Google Colab by following this link: [Google Colab Link](https://colab.research.google.com/drive/19T5O_GintvH6FB-X6VdfCd1rBpKJ1eEe#scrollTo=TD-ZwCEM3PqY)

### Steps to Run in Google Colab

1. **Clone the repository to Google Colab:**

    ```python
    !git clone https://github.com/Rechef-Team/rechef-ml.git
    ```

2. **Change to the repository directory:**

    ```python
    %cd rechef-ml
    ```

3. **Install the required packages:**

    ```python
    !pip install tensorflow==2.13.0 keras==2.13.1 matplotlib==3.9.0 numpy==1.24.3 scikit-learn==1.5.0 seaborn==0.13.2
    ```

4. **Download and unzip the dataset:**

    ```python
    !wget https://github.com/Rechef-Team/rechef-ml/blob/main/DatasetBahanV1.zip
    !unzip DatasetBahanV1.zip -d ./data
    ```

### Example Code to Run the Model

Here is an example of how to run the model after setting up your environment:

```python
# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data preprocessing
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory('./data', target_size=(150, 150), batch_size=32, class_mode='categorical', subset='training')
validation_data = data_gen.flow_from_directory('./data', target_size=(150, 150), batch_size=32, class_mode='categorical', subset='validation')

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')  # 8 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
model.fit(train_data, validation_data=validation_data, epochs=10)
```

Pastikan Anda mengganti path `./data` dengan path yang sesuai jika berbeda. Contoh ini menyediakan arsitektur jaringan neural convolutional dasar untuk klasifikasi bahan makanan, tetapi Anda mungkin perlu menyesuaikannya berdasarkan kebutuhan dan dataset spesifik Anda.
