import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Tentukan path model lokal
model_path = 'D:\\Re Chef\\Codingan\\rechefmodel.h5'

# Cek apakah file model ada
if not os.path.exists(model_path):
    raise OSError(f"Tidak ditemukan file atau direktori di {model_path}")

# Muat model yang telah dilatih
model = tf.keras.models.load_model(model_path)

# Nama-nama kelas
class_names = ['BAWANG MERAH', 'BAWANG PUTIH', 'CABAI', 'DAGING AYAM', 'TAHU', 'TEMPE', 'TELUR', 'TOMAT']  # Ganti dengan nama kelas yang sebenarnya

# Fungsi untuk resize gambar
def custom_augment(image):
    return tf.image.resize(image, [224, 224])

# Preproses gambar dengan resize
def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# Prediksi dan cek bahan makanan
def recommend_food(image_path, model, class_names, threshold=0.8):
    image = preprocess_image(image_path, (224, 224))
    predictions = model.predict(image)
    max_prob = np.max(predictions)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class < len(class_names):
        predicted_class_name = class_names[predicted_class]
    else:
        predicted_class_name = "Unknown class"

    if max_prob >= threshold:
        result = f"Nama: {predicted_class_name}, Kemiripan: {max_prob*100:.2f}%"
    else:
        result = "Invalid atau tidak bahan makanan"

    plt.imshow(load_img(image_path))
    plt.title(result)
    plt.axis('off')
    plt.show()

    return result

# Fungsi utama untuk merekomendasikan bahan makanan
def main():
    root = Tk()
    root.withdraw()  # Sembunyikan jendela root
    image_paths = filedialog.askopenfilenames(
        title="Pilih Gambar",
        filetypes=[("File gambar", "*.jpg *.jpeg *.png")]
    )

    for image_path in image_paths:
        result = recommend_food(image_path, model, class_names)
        print(f"Hasil untuk {os.path.basename(image_path)}: {result}")

if __name__ == '__main__':
    main()
