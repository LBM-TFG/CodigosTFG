import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Título de la aplicación
st.title("Ejemplo de captura de vídeo con Streamlit-WebRTC para la aplicaciónweb")

# Widget para la transmisión de video en tiempo real
webrtc_streamer(
    key="example",              # Identificador único
    video_frame_callback=None,  # Sin procesamiento adicional (solo muestra el video)
    media_stream_constraints={  # Configuración de la cámara
        "video": True,          # Habilita video
        "audio": False          # Deshabilita audio
    },
    async_processing=False
)

