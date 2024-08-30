"""
Author: Cid Emmanuel Esquivel González
Github: https://github.com/CidESQ/Filtrado-de-Im-genes-en-Python
Topic: Tarea 1 Vision Artificial
"""
# Librerias
from tkinter import filedialog
import cv2
import numpy as np

# Abrir el cuadro de dialogo para seleccionar una imagen
image_path = filedialog.askopenfilename(title='Seleccione la imagen')
imagen = cv2.imread(image_path)  # Leer imagen seleccionada
imagen_principal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Separar los canales RGB
R, G, B = cv2.split(imagen_principal)

'''
 CREACION DE KERNELS/FILTROS
'''
kernel_1 = np.ones((3, 3), np.float32) / 9  # Blur / media
kernel_2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16  # Gausiano
kernel_3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # Realce / Sharpening
kernel_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)  # Laplaciano / Bordes en todas direcciones
kernel_5 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)  # Filtro Sobel Horizontal
# Convolcion de kernels: kernel_combinado = convolve2d(kernel_a, kernel_b, mode='full')

# Funcion
def aplicar_kernel(kernel):
    r_kernel = cv2.filter2D(R, -1, kernel)
    g_kernel = cv2.filter2D(G, -1, kernel)
    b_kernel = cv2.filter2D(B, -1, kernel)
    imagen_filtrada_rgb = cv2.merge((r_kernel, g_kernel, b_kernel))
    # imagen_filtrada_bgr = cv2.cvtColor(imagen_filtrada_rgb, cv2.COLOR_RGB2BGR)  # mostrar con OpenCV
    return imagen_filtrada_rgb


def interfaz_2():
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Filtro Blur', cv2.cvtColor(aplicar_kernel(kernel_1), cv2.COLOR_RGB2BGR))
    cv2.imshow('Filtro Gausiano', cv2.cvtColor(aplicar_kernel(kernel_2), cv2.COLOR_RGB2BGR))
    cv2.imshow('Sharpening', cv2.cvtColor(aplicar_kernel(kernel_3), cv2.COLOR_RGB2BGR))
    cv2.imshow('Laplaciano', cv2.cvtColor(aplicar_kernel(kernel_4), cv2.COLOR_RGB2BGR))
    cv2.imshow('Sobel', cv2.cvtColor(aplicar_kernel(kernel_5), cv2.COLOR_RGB2BGR))
    # Con el teclado pasamos a la imagen
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# interfaz_1()
interfaz_2()

'''
Explicacion filtros/kernels

1- Blur: Suaviza la imagen promediando los valores de los píxeles vecinos. Esto reduce el ruido y los detalles finos, 
    produciendo un efecto de desenfoque.
    
2- Gaussiano: También se usa para suavizar la imagen, pero de una manera que preserva mejor los bordes comparado con el
    blur simple. El filtro gaussiano reduce el ruido mientras mantiene la estructura de la imagen más nítida.
    
3- Sharpening: Aumenta el contraste y la nitidez de la imagen, destacando los detalles y bordes.

4- Laplaciano: Resalta los bordes y las áreas de transición en la imagen al calcular la segunda derivada de la
    intensidad de los píxeles.
    
5- Sobel: está diseñado para resaltar los cambios en la intensidad de la imagen que ocurren a lo largo del eje
    horizontal. En términos más simples, detecta bordes que son verticales en la imagen.
'''
