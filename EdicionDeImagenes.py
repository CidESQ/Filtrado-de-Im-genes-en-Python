# Librerias
import tkinter as tk
from tkinter import filedialog
import matplotlib

matplotlib.use('MacOSX')

from PIL import Image
import pillow_heif
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Abrir una imagen
# root = tk.Tk()
# root.withdraw()

# Abrir el cuadro de dialogo para seleccionar un arhcivo
image_path = filedialog.askopenfilename(
    title='Seleccione la imagen'
)

# Leer imagen seleccionada
imagen = cv2.imread(image_path)
# size = imagen.shape
# cv2.imshow("Imagen escogida", imagen)
imagen_principal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Separar los canales RGB
R, G, B = cv2.split(imagen_principal)

# creacion de filtro 1 Blur
kernel_1 = np.ones((13, 13), np.float32) / 169

# Aplicar el filtro de media a cada canal
R_blur = cv2.filter2D(R, -1, kernel_1)
G_blur = cv2.filter2D(G, -1, kernel_1)
B_blur = cv2.filter2D(B, -1, kernel_1)

# Unir los canales filtrados para crear la imagen final
imagen_filtro_1 = cv2.merge((R_blur, G_blur, B_blur))
imagen_filtro_1_bgr = cv2.cvtColor(imagen_filtro_1, cv2.COLOR_RGB2BGR)
cv2.imshow("Imagen original", imagen)
cv2.imshow("Imagen Filtrada (blur)", imagen_filtro_1_bgr)

# Interfaz Grafica
# figura = plt.figure()
# # Imagen Principal
# ax1 = figura.add_subplot(3, 3, 1)
# ax1.imshow(imagen_principal, cmap="gray")
# ax1.set_title("Imagen Principal")
#
# # Filtro 1
# ax2 = figura.add_subplot(3, 3, 2)
# ax2.imshow(imagen_filtro_1, cmap="gray")
# ax2.set_title("Filtro 1")
#
# # Filtro 2
# ax3 = figura.add_subplot(3, 3, 3)
# ax3.imshow(imagen_filtro_2, cmap="gray")
# ax3.set_title("Filtro 2")
#
# # Filtro 3
# ax4 = figura.add_subplot(3, 3, 4)
# ax4.imshow(imagen_filtro_3, cmap="gray")
# ax4.set_title("Filtro 3")
#
# # Filtro 4
# ax5 = figura.add_subplot(3, 3, 5)
# ax5.imshow(imagen_filtro_4, cmap="gray")
# ax5.set_title("Filtro 4")
#
# # Filtro 5
# ax6 = figura.add_subplot(3, 3, 6)
# ax6.imshow(imagen_filtro_5, cmap="gray")
# ax6.set_title("Filtro 5")

# plt.show()

# Con el teclado pasamos a la imagen
cv2.waitKey(0)
cv2.destroyAllWindows()
