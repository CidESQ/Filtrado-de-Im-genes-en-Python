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

