"""
Author: Cid Emmanuel Esquivel González
Github: https://github.com/CidESQ/Filtrado-de-Im-genes-en-Python
Topic: Tarea 1 Vision Artificial
Instrucciones:
    ~ Ejecutar el programa
    ~ Seleccionar imagen
    ~ Ver las diferentes ventanas con cada filtro
    ~ Terminar -> Presionar tecla '0'
"""
# Librerias
from tkinter import filedialog
import cv2
import numpy as np

# Abrir el cuadro de dialogo para seleccionar una imagen
image_path = filedialog.askopenfilename(title='Seleccione la imagen')
original_image = cv2.imread(image_path)  # Leer imagen seleccionada

# Separar los canales RGB (BGR en OpenCV) para aplicar la convolucion en cada uno y despues volverlos a unir.
B, G, R = cv2.split(original_image)
'''
 DEFINICION DE KERNELS/FILTROS
'''
kernel_1 = np.ones((3, 3), np.float32) / 9  # Blur / media
kernel_2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16  # Gausiano
kernel_3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Realce / Sharpening
kernel_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)  # Laplaciano / Bordes en todas direcciones
kernel_5 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)  # Filtro Emboss
kernel_6 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)  # Filtro High-Pass
kernel_7 = np.array([[1, 0], [0, -1]], np.float32)  # Filtro Robert Cross
# Convolcion de kernels: kernel_combinado = convolve2d(kernel_a, kernel_b, mode='full')


def apply_kernel(kernel):
    b_kernel = cv2.filter2D(B, -1, kernel)  # ? aplicar un filtro de convolución a un canal
    g_kernel = cv2.filter2D(G, -1, kernel)
    r_kernel = cv2.filter2D(R, -1, kernel)
    filtered_image = cv2.merge((b_kernel, g_kernel, r_kernel))
    return filtered_image


def show_filters():
    cv2.imshow('Imagen Original', original_image)
    cv2.imshow('1 Filtro Blur', apply_kernel(kernel_1))
    cv2.imshow('2 Filtro Gausiano', apply_kernel(kernel_2))
    cv2.imshow('3 Sharpening', apply_kernel(kernel_3))
    cv2.imshow('4 Laplaciano', apply_kernel(kernel_4))
    cv2.imshow('5 Emboss', apply_kernel(kernel_5))
    cv2.imshow('6 High pass', apply_kernel(kernel_6))
    cv2.imshow('7 Robert Cross', apply_kernel(kernel_7))
    cv2.waitKey(0)  # ? Terminar programa con tecla '0'
    cv2.destroyAllWindows()


show_filters()

'''
Explicacion filtros/kernels

1- Blur: Suaviza la imagen promediando los valores de los píxeles vecinos. Esto reduce el ruido y los detalles finos, 
    produciendo un efecto de desenfoque.
    
2- Gaussiano: También se usa para suavizar la imagen, pero de una manera que preserva mejor los bordes comparado con el
    blur simple. El filtro gaussiano reduce el ruido mientras mantiene la estructura de la imagen más nítida.
    
3- Sharpening: Aumenta el contraste y la nitidez de la imagen, destacando los detalles y bordes.

4- Laplaciano: Resalta los bordes y las áreas de transición en la imagen al calcular la segunda derivada de la
    intensidad de los píxeles.
    
5- Emboss: está diseñado para crear un efecto de relieve que resalta los bordes en la imagen, dándole un aspecto
    tridimensional.

6- High-Pass: resalta los detalles finos en la imagen, eliminando las frecuencias bajas (como áreas suaves).

7- Robert cross:  Es un filtro de detección de bordes que resalta los bordes en una diagonal particular, 
    utilizado comúnmente para detectar cambios rápidos en la intensidad de la imagen.
'''
