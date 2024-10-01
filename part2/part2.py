import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger les modèles pré-entraînés
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# Classes d'âge et de genre
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Charger l'image
frame = cv2.imread('image2.jpg')
h, w = frame.shape[:2]

# Prétraitement
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Détection de visage
face_net.setInput(blob)
detections = face_net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Extraction du visage
        face = frame[startY:endY, startX:endX]
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))

        # Prédiction du genre
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Prédiction de l'âge
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Affichage des résultats
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Sauvegarder l'image avec les prédictions
cv2.imwrite('face_detection_result.jpg', frame)

# Afficher l'image avec matplotlib (au lieu de cv2.imshow)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convertir BGR vers RGB pour matplotlib
plt.title('Détection des visages')
plt.axis('off')  # Masquer les axes
plt.show()

# Optionnel : Attendre une touche pour fermer les fenêtres (si nécessaire)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
