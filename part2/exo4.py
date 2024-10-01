import cv2

# Créer le tracker CSRT
tracker = cv2.TrackerCSRT_create()

# Capturer la vidéo
cap = cv2.VideoCapture(0)

# Lire la première image et permettre à l'utilisateur de sélectionner l'objet à suivre
ret, frame = cap.read()
bbox = cv2.selectROI('Sélectionnez l\'objet à suivre', frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mettre à jour la position de l'objet
    ret, bbox = tracker.update(frame)

    if ret:
        # Dessiner le rectangle autour de l'objet
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Objet perdu', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow('Suivi d\'objet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
