import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Función original para contar bíceps
def contar_biceps(frame_rgb, results, width, height, estado, contando):
    if results.pose_landmarks:
        biceps_izq = [(int(results.pose_landmarks.landmark[i].x * width),
                       int(results.pose_landmarks.landmark[i].y * height)) for i in [12, 14, 16]]
        biceps_der = [(int(results.pose_landmarks.landmark[i].x * width),
                       int(results.pose_landmarks.landmark[i].y * height)) for i in [11, 13, 15]]

        def biceps_calcular_angulo(a, b, c):
            biceps_ba = np.array(a) - np.array(b)
            biceps_bc = np.array(c) - np.array(b)
            biceps_coseno_angulo = np.dot(biceps_ba, biceps_bc) / (np.linalg.norm(biceps_ba) * np.linalg.norm(biceps_bc))
            return degrees(acos(biceps_coseno_angulo))

        biceps_angulo_der = biceps_calcular_angulo(*biceps_der)
        biceps_angulo_izq = biceps_calcular_angulo(*biceps_izq)

        def actualizar_contador(lado, angulo):
            abierto = f'abierto_{lado}'
            cerrado = f'cerrado_{lado}'
            if angulo >= 150:
                estado[abierto] = True
            if estado[abierto] and not estado[cerrado] and angulo <= 50:
                estado[cerrado] = True
            if estado[abierto] and estado[cerrado] and angulo >= 150:
                estado[lado] += 1
                estado['tot'] += 1
                estado[abierto] = False
                estado[cerrado] = False
                return True
            return False

        if contando:
            if actualizar_contador('der', biceps_angulo_der) or actualizar_contador('izq', biceps_angulo_izq):
                cv2.putText(frame_rgb, "REP COMPLETA!", (width//2-100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        color_der = (0, 255, 0) if contando else (200, 200, 200)
        color_izq = (0, 255, 0) if contando else (200, 200, 200)

        for i in range(2):
            cv2.line(frame_rgb, biceps_der[i], biceps_der[i+1], color_der, 3)
            cv2.line(frame_rgb, biceps_izq[i], biceps_izq[i+1], color_izq, 3)

        cv2.putText(frame_rgb, f"D: {int(biceps_angulo_der)}°", (biceps_der[1][0], biceps_der[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_rgb, f"I: {int(biceps_angulo_izq)}°", (biceps_izq[1][0], biceps_izq[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_rgb, estado

# Procesador
class BicepsProcessor(VideoProcessorBase):
    def __init__(self):
        self.estado = {
            'tot': 0, 'der': 0, 'izq': 0,
            'abierto_der': False, 'cerrado_der': False,
            'abierto_izq': False, 'cerrado_izq': False
        }
        self.contando = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        img, self.estado = contar_biceps(img, results, img.shape[1], img.shape[0], self.estado, self.contando)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Interfaz
st.title("💪 Contador de Bíceps con WebRTC")

if 'processor' not in st.session_state:
    st.session_state.processor = None

col1, col2 = st.columns([1, 2])
with col1:
    start_stop = st.button("Iniciar / Detener")
    reset = st.button("Resetear Contadores")
    if st.session_state.processor:
        if start_stop:
            st.session_state.processor.contando = not st.session_state.processor.contando
        if reset:
            st.session_state.processor.estado = {
                'tot': 0, 'der': 0, 'izq': 0,
                'abierto_der': False, 'cerrado_der': False,
                'abierto_izq': False, 'cerrado_izq': False
            }

with col2:
    if st.session_state.processor:
        st.markdown(f"""
        ### 📊 Contadores
        - **Derecho:** `{st.session_state.processor.estado['der']}`
        - **Izquierdo:** `{st.session_state.processor.estado['izq']}`
        - **Total:** `{st.session_state.processor.estado['tot']}`
        """)

ctx = webrtc_streamer(
    key="biceps",
    video_processor_factory=BicepsProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Guardar referencia al procesador para control externo
if ctx.video_processor:
    st.session_state.processor = ctx.video_processor
