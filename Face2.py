import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Charger le classificateur de visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default_1.xml')

# Fonction de détection
def detect_faces(scaleFactor, minNeighbors, rectangle_color):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la webcam.")
        return

    stframe = st.empty()  # Conteneur pour afficher les images dans Streamlit

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Erreur : Impossible de lire une image depuis la webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Convertir l'image en RGB pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Enregistrer l'image si demandé
        if st.session_state.get("save_requested", False):
            filename = f"detected_faces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Image enregistrée sous : {filename}")
            st.session_state.save_requested = False  # Réinitialiser

        # Gérer l'arrêt avec un bouton dans Streamlit
        if st.session_state.get("stop_requested", False):
            break

    cap.release()

# Application Streamlit
def app():
    st.title("Détection de visages avec OpenCV et Streamlit")

    # Gestion d'état
    if "save_requested" not in st.session_state:
        st.session_state.save_requested = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    # Paramètres personnalisables
    st.sidebar.title("Paramètres")
    scaleFactor = st.sidebar.slider("Facteur de redimensionnement", 1.0, 2.0, 1.3, step=0.1)
    minNeighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, step=1)
    color = st.sidebar.color_picker("Couleur des rectangles", "#00FF00")
    rectangle_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    # Boutons
    if st.button("Démarrer la détection"):
        detect_faces(scaleFactor, minNeighbors, rectangle_color)

    if st.button("Enregistrer l'image"):
        st.session_state.save_requested = True

    if st.button("Arrêter la détection"):
        st.session_state.stop_requested = True

if __name__ == "__main__":
    app()
