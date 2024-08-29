# Librerias
import tkinter as tk
from tkinter import filedialog

from PIL import Image
import pillow_heif
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Abrir una imagen
root = tk.Tk()
root.withdraw()

# Abrir el cuadro de dialogo para seleccionar un arhcivo
image_path = filedialog.askopenfilename(
    title='Seleccione la imagen'
)

# Separar los canales RGB
original_image = Image.open(image_path)  # Cargar la imagen .jpg
image_np = np.array(original_image)  # Covnertir la imagen a un array NumPy
R = image_np[:, :, 0]
G = image_np[:, :, 1]
B = image_np[:, :, 2]


