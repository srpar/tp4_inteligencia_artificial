import cv2 # Se importa la libreria OpenCV para el procesamiento de imagenes
import numpy as np # Se importa la libreria numpy con el alias np. Aporta soporte para matrices y diversas funciones matemáticas 
import matplotlib.pyplot as plt # Se importa un modulo de la libreria pyplot con un alias que permite generar graficos
from scipy.spatial import distance # Se importa un modulo de la libreria scipy para facilitar calculo de distancias en espacios multidimensionales

# Se carga la imagen del caso del aro en la que se debe identificar la ubicacion en coordenadas del centro del aro
# Metodo de openCV para leer la imagen
imagen = cv2.imread('frente-motor.jpg')

# Se convierte la imagen cargada a escala de grises para facilitar el procesamiento
# Se aplica un filtro de tipo gausiano a la imagen
# Se utiliza la funcion Canny de OpenCV para detectar bordes de la imagen ya procesada a escala de gris y suavizada con el filtro Gausiano
imagen_escala_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
suavizado = cv2.GaussianBlur(imagen_escala_grises, (7, 7), 0)
bordes = cv2.Canny(suavizado, 50, 150)

# Se aplica  la transformada de Hough para detectar circulos
# Los parametros determinados permiten focalizar en el circulo mas importante para el caso puntual
circulo = cv2.HoughCircles(suavizado, cv2.HOUGH_GRADIENT, 1, 280)

# Se dibuja el principal círculo en la imagen original
if circulo is not None:
    circulo = np.round(circulo[0, :]).astype("int")
    for (x, y, r) in circulo:
        cv2.circle(imagen, (x, y), r, (0, 255, 255), 4)

# Se muestra la imagen en escala de grises
# Luego se muestra la imagen con sus lineas y circulos

cv2.imshow('Imagen que incluye el circulo principal', imagen_escala_grises)
cv2.waitKey(0)
cv2.imshow('Imagen que incluye el circulo principal', imagen)
cv2.waitKey(0)


# Se muestra a cada lado la imagen con los circulos, y la imagen procesada que se toma de base
# Se utiliza otra libreria (matplotlib.pyplot) que tiene otra presentacion diferente respecto a OpenCv
plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen que incluye el circulo principal')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(bordes, cv2.COLOR_BGR2RGB))
plt.title('Imagen procesada a la cual se le detecta el circulo principal')

plt.show()

# Finalmente, luego de presionar una tecla se cierra la ventana emergente abierta
cv2.destroyAllWindows() 