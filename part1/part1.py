import cv2
import numpy as np
from matplotlib import pyplot as plt

# Partie 1 exo 1

# Charger l'image en haute résolution
image = cv2.imread('image2.jpg', 0)

# Appliquer le filtre Sobel pour les contours
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)

# Afficher les résultats
cv2.imshow('Contours Sobel', sobel_combined)
cv2.imwrite('sobel_result.jpg', sobel_combined)

# Transformation de Fourier
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Manipulation du spectre et transformation inverse
rows, cols = image.shape
crow, ccol = rows//2 , cols//2
dft_shift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# Afficher le spectre modifié
cv2.imshow('Transformation Inverse', img_back)
cv2.imwrite('fourier_result.jpg', img_back)

# Seuillage adaptatif
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Afficher les résultats
cv2.imshow('Segmentation par seuillage adaptatif', thresh)
cv2.imwrite('segmentation_result.jpg', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Partie 1 exo 2

# Charger l'image
image = cv2.imread('image2.jpg')

# Sélection de la région d'intérêt (ROI)
roi = image[100:300, 150:350]  # Par exemple, un rectangle de 200x200

# Appliquer une modification à la ROI (augmentation de la luminosité)
roi_brightened = cv2.convertScaleAbs(roi, alpha=1.5, beta=50)

# Remplacer la région d'intérêt dans l'image originale
image[100:300, 150:350] = roi_brightened

# Appliquer un flou gaussien sur le reste de l'image
mask = np.zeros_like(image)
mask[100:300, 150:350] = 1
blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
final_image = np.where(mask == 1, image, blurred_image)

# Afficher et sauvegarder l'image
cv2.imshow('Image avec ROI', final_image)
cv2.imwrite('roi_result.jpg', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
