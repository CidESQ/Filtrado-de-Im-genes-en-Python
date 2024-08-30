# Librerias
from tkinter import filedialog
import matplotlib
import cv2
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Abrir el cuadro de dialogo para seleccionar una imagen
image_path = filedialog.askopenfilename(title='Seleccione la imagen')
# Leer imagen seleccionada
imagen = cv2.imread(image_path)
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
kernel_5 = np.array([[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], np.float32)  # Filtro Prewitt horizontal


# Funcion
def aplicar_kernel(kernel):
    R_kernel = cv2.filter2D(R, -1, kernel)
    G_kernel = cv2.filter2D(G, -1, kernel)
    B_kernel = cv2.filter2D(B, -1, kernel)
    imagen_filtrada_rgb = cv2.merge((R_kernel, G_kernel, B_kernel))
    # imagen_filtrada_bgr = cv2.cvtColor(imagen_filtrada_rgb, cv2.COLOR_RGB2BGR)  # mostrar con OpenCV
    return imagen_filtrada_rgb


# Interfaz Gráfica
figura = plt.figure()
# Imagen Principal
ax1 = figura.add_subplot(3, 3, 1)
ax1.imshow(imagen_principal)
ax1.set_title("Imagen Principal")

# Filtro 1
ax2 = figura.add_subplot(3, 3, 2)
ax2.imshow(aplicar_kernel(kernel_1))
ax2.set_title("Filtro 1")

# Filtro 2
ax3 = figura.add_subplot(3, 3, 3)
ax3.imshow(aplicar_kernel(kernel_2))
ax3.set_title("Filtro 2")

# Filtro 3
ax4 = figura.add_subplot(3, 3, 4)
ax4.imshow(aplicar_kernel(kernel_3))
ax4.set_title("Filtro 3")

# Filtro 4
ax5 = figura.add_subplot(3, 3, 5)
ax5.imshow(aplicar_kernel(kernel_4))
ax5.set_title("Filtro 4")

# Filtro 5
ax6 = figura.add_subplot(3, 3, 6)
ax6.imshow(aplicar_kernel(kernel_5))
ax6.set_title("Filtro 5")

plt.show()

# Con el teclado pasamos a la imagen
# cv2.waitKey(0)
# cv2.destroyAllWindows()
