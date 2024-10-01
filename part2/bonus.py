import cv2
import mediapipe as mp
import numpy as np

# Initialisation de MediaPipe pour la détection de mains
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration du détecteur de mains MediaPipe
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Capturer la vidéo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en RGB (MediaPipe fonctionne avec des images RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dessiner les points de repère des mains
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Reconnaissance de gestes simples en utilisant la position des points de repère
            # Par exemple, comparer la distance entre certains points pour différencier des gestes
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convertir les coordonnées relatives à l'image en pixels
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            middle_x, middle_y = int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])

            # Exemple de logique pour reconnaître une "paume ouverte" ou un "poing fermé"
            if abs(thumb_y - index_y) > 100:  # Par exemple, si le pouce est éloigné de l'index
                gesture = "Paume ouverte"
            else:
                gesture = "Poing fermé"

            # Afficher le geste détecté
            cv2.putText(frame, gesture, (thumb_x, thumb_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher l'image avec les mains et les gestes détectés
    cv2.imshow('Détection de mains et gestes avec MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
