# Aplicacion de Lucia Bayo Monedero

########################################################################################################################################################
# Librerias --------------------------------------------------------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import time

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock

########################################################################################################################################################
# Inicializar Mediapipe --------------------------------------------------------------------------------------------------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils


########################################################################################################################################################
# Variables --------------------------------------------------------------------------------------------------------------------------------------------

# Biceps
bicepsAbiertoDer = False
bicepsCerradoDer = False
bicepsAbiertoIzq = False
bicepsCerradoIzq = False
bicepsContadorDer = 0
bicepsContadorIzq = 0
bicepsContadorTot = 0

# Flexiones
flexionesArriba = False
flexionesAbajo = False
flexionesMedioBajando = False
flexionesMedioSubiendo = False
flexionesSubidaCompleta = False
flexionesBajadaCompleta = False
flexionesContador = 0
flexionesContadorNoCompleta = 0

# Sentadillas
sentadillasArriba = False
sentadillasAbajo = False
sentadillasMedioBajando = False
sentadillasMedioSubiendo = False
sentadillasSubidaCompleta = False
sentadillasBajadaCompleta = False
sentadillasContador = 0
sentadillasContadorNoCompleta = 0

# Zancada atrás
abiertoDer = False
cerradoDer = False
abiertoIzq = False
cerradoIzq = False
zancadaContadorIzq = 0
zancadaContadorDer = 0
zancadaContador = 0

# Crunch abdominal
abdominalesEstirado = False
abdominalesContraido = False
abdominalesContador = 0

# Dominadas
dominadasArriba = False
dominadasAbajo = False
dominadasMedioBajando = False
dominadasMedioSubiendo = False
dominadasSubidaCompleta = False
dominadasBajadaCompleta = False
dominadasContador = 0
dominadasContadorNoCompleta = 0

# Otros
start_time = time.time()            # Temporizador
cap = None                          # Videocaptura
ejercicio_seleccionado = None       # Sin ejercicio por defecto



########################################################################################################################################################
# Botones  --------------------------------------------------------------------------------------------------------------------------------------------

class MainLayout(BoxLayout):
    # Funciones para los botones
    def btnIniciar_click(self):
        # Inicializa la captura de video
        global cap
        #cap = cv2.VideoCapture("Zatras.mp4")
        cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.visualizar, 1.0 / 30.0)  # 30 FPS
        print("¡Botón INICIAR presionado!")

    def btnTerminar_click(self):
        global cap
        if cap is not None:  # Solo detiene la cámara si está activa
            cap.release()
            cap = None
            Clock.unschedule(self.visualizar)
        print("¡Botón TERMINAR presionado!")

    def btnBiceps_click(self):
        print("¡Botón BICEPS presionado!")
        self.cambiar_ejercicio("Biceps")

    def btnFlexiones_click(self):
        print("¡Botón FLEXIONES presionado!")
        self.cambiar_ejercicio("Flexiones")

    def btnSentadillas_click(self):
        print("¡Botón SENTADILLAS presionado!")
        self.cambiar_ejercicio("Sentadillas")

    def btnZancadaAtras_click(self):
        print("¡Botón ZANCADAATRAS presionado!")
        self.cambiar_ejercicio("Zancada atras")

    def btnCrunchAbdominal_click(self):
        print("¡Botón CRUNCHABDOMINAL presionado!")
        self.cambiar_ejercicio("Crunch abdominal")

    def btnDominadas_click(self):
        print("¡Botón DOMINADAS presionado!")
        self.cambiar_ejercicio("Dominadas")

    def cambiar_ejercicio(self, ejercicio):
        global ejercicio_seleccionado
        ejercicio_seleccionado = ejercicio
        print(f"Ejercicio seleccionado: {ejercicio}")
        self.mostrar_popup(ejercicio)


 # Función para mostrar ventana emergente personalizada
    def mostrar_popup(self, ejercicio):
        # Diccionario con imágenes y mensajes personalizados
        info_ejercicios = {
            "Biceps": {"imagen": "biceps.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\nLos ejercicios de bíceps sirven para fortalecer los brazos."},
            "Flexiones": {"imagen": "flexiones.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\nLas flexiones trabajan el pecho y los tríceps."},
            "Sentadillas": {"imagen": "sentadillas.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\nLas sentadillas fortalecen las piernas ylos glúteos."},
            "Zancada atras": {"imagen": "zancada.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\n Las zancada atrás sirven para mejorar el equilibrio y la fuerza de las piernas."},
            "Crunch abdominal": {"imagen": "abdominales.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\nEste ejercicio sirve para fortalecer los abdominales."},
            "Dominadas": {"imagen": "dominadas.png", "mensaje": "Recuerda mantener una postura correcta al realizar el ejercicio.\nLas dominadas fortalecen la espalda y los brazos."}
        }

        # Manejar el caso en que no haya ejercicio seleccionado
        if ejercicio is None:
            datos = {"imagen": "default.png", "mensaje": "Por favor, selecciona un ejercicio."}
        else:
            datos = info_ejercicios.get(ejercicio, {"imagen": "default.png", "mensaje": "Ejercicio no definido."})

        # Obtener datos del ejercicio o usar valores predeterminados
        #datos = info_ejercicios.get(ejercicio, {"imagen": "default.png", "mensaje": "Ejercicio no definido."})
        
        # Crear diseño de la ventana emergente
        contenido = BoxLayout(orientation='vertical', spacing=10)
        contenido.add_widget(Image(source=datos["imagen"]))  # Agregar imagen
        contenido.add_widget(Label(text=datos["mensaje"], font_size=18))  # Agregar mensaje
        
        # Crear y abrir el popup
        popup = Popup(title=f"Ejercicio actual: {ejercicio}",
            content=contenido,
            size_hint=(None, None), size=(700, 400))
        popup.open()



########################################################################################################################################################
# Curl de Bíceps --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_biceps(self, frame_rgb, results, width, height):
        global bicepsContadorTot, bicepsContadorDer, bicepsContadorIzq, bicepsAbiertoDer, bicepsCerradoDer, bicepsAbiertoIzq, bicepsCerradoIzq

        # Coordenadas de los brazos
        bicepsX1_der = int(results.pose_landmarks.landmark[12].x * width)   # Punto 12: hombro derecho
        bicepsY1_der = int(results.pose_landmarks.landmark[12].y * height)  
        bicepsX2_der = int(results.pose_landmarks.landmark[14].x * width)   # Punto 14: codo derecho
        bicepsY2_der = int(results.pose_landmarks.landmark[14].y * height)  
        bicepsX3_der = int(results.pose_landmarks.landmark[16].x * width)   # Punto 16: muñeca derecha
        bicepsY3_der = int(results.pose_landmarks.landmark[16].y * height)  

        bicepsX1_izq = int(results.pose_landmarks.landmark[11].x * width)   # Punto 11: hombro izquierdo
        bicepsY1_izq = int(results.pose_landmarks.landmark[11].y * height)  
        bicepsX2_izq = int(results.pose_landmarks.landmark[13].x * width)   # Punto 13: codo izquierdo
        bicepsY2_izq = int(results.pose_landmarks.landmark[13].y * height)  
        bicepsX3_izq = int(results.pose_landmarks.landmark[15].x * width)   # Punto 15: muñeca izquierda
        bicepsY3_izq = int(results.pose_landmarks.landmark[15].y * height)  

        # Definir las localizaciones de los puntos de referencia
        bicepsPos1_der = np.array([bicepsX1_der, bicepsY1_der])
        bicepsPos2_der = np.array([bicepsX2_der, bicepsY2_der])
        bicepsPos3_der = np.array([bicepsX3_der, bicepsY3_der])

        bicepsPos1_izq = np.array([bicepsX1_izq, bicepsY1_izq])
        bicepsPos2_izq = np.array([bicepsX2_izq, bicepsY2_izq])
        bicepsPos3_izq = np.array([bicepsX3_izq, bicepsY3_izq])
        
        # Cálculo de los lados del triángulo
        bicepsLado1_der = np.linalg.norm(bicepsPos2_der - bicepsPos3_der)
        bicepsLado2_der = np.linalg.norm(bicepsPos1_der - bicepsPos3_der)
        bicepsLado3_der = np.linalg.norm(bicepsPos1_der - bicepsPos2_der)
        
        bicepsLado1_izq = np.linalg.norm(bicepsPos2_izq - bicepsPos3_izq)
        bicepsLado2_izq = np.linalg.norm(bicepsPos1_izq - bicepsPos3_izq)
        bicepsLado3_izq = np.linalg.norm(bicepsPos1_izq - bicepsPos2_izq)

        # Cálculo del ángulo
        bicepsAngulo_der = degrees(acos((bicepsLado1_der**2 + bicepsLado3_der**2 - bicepsLado2_der**2) / (2 * bicepsLado1_der * bicepsLado3_der)))
        bicepsAngulo_izq = degrees(acos((bicepsLado1_izq**2 + bicepsLado3_izq**2 - bicepsLado2_izq**2) / (2 * bicepsLado1_izq * bicepsLado3_izq)))

        # Condicionantes del ejercicio
        if bicepsAngulo_der >= 150:
            bicepsAbiertoDer = True
        if bicepsAbiertoDer == True and bicepsCerradoDer == False and bicepsAngulo_der <= 50:
            bicepsCerradoDer = True
        if bicepsAbiertoDer == True and bicepsCerradoDer == True and bicepsAngulo_der >= 150:
            bicepsContadorDer += 1
            bicepsAbiertoDer = False
            bicepsCerradoDer = False

        if bicepsAngulo_izq >= 150:
            bicepsAbiertoIzq = True
        if bicepsAbiertoIzq == True and bicepsCerradoIzq == False and bicepsAngulo_izq <= 50:
            bicepsCerradoIzq = True
        if bicepsAbiertoIzq == True and bicepsCerradoIzq == True and bicepsAngulo_izq >= 150:
            bicepsContadorIzq += 1
            bicepsAbiertoIzq = False
            bicepsCerradoIzq = False

        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(bicepsAngulo_der), (bicepsX2_der, bicepsY2_der - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)
        cv2.putText(frame_rgb, "{:.2f}".format(bicepsAngulo_izq), (bicepsX2_izq, bicepsY2_izq - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (bicepsX1_der, bicepsY1_der), (bicepsX2_der, bicepsY2_der), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (bicepsX2_der, bicepsY2_der), (bicepsX3_der, bicepsY3_der), (0, 255, 0), 3)  # Codo a muñeca
        cv2.line(frame_rgb, (bicepsX1_izq, bicepsY1_izq), (bicepsX2_izq, bicepsY2_izq), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (bicepsX2_izq, bicepsY2_izq), (bicepsX3_izq, bicepsY3_izq), (0, 255, 0), 3)  # Codo a muñeca
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_biceps_der.text = str(bicepsContadorDer)
        self.ids.etiq_contador_biceps_izq.text = str(bicepsContadorIzq)
        self.ids.etiq_contador_biceps_tot.text = str(bicepsContadorDer + bicepsContadorIzq)

        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)


########################################################################################################################################################
# Flexiones --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_flexiones(self, frame_rgb, results, width, height):
        global flexionesContador, flexionesContadorNoCompleta, flexionesArriba, flexionesAbajo, flexionesMedioBajando, flexionesMedioSubiendo, flexionesSubidaCompleta, flexionesBajadaCompleta

        # Coordenadas de los brazos
        flexX1_der = int(results.pose_landmarks.landmark[12].x * width)   # Punto 12: hombro derecho
        flexY1_der = int(results.pose_landmarks.landmark[12].y * height)  
        flexX2_der = int(results.pose_landmarks.landmark[14].x * width)   # Punto 14: codo derecho
        flexY2_der = int(results.pose_landmarks.landmark[14].y * height)  
        flexX3_der = int(results.pose_landmarks.landmark[16].x * width)   # Punto 16: muñeca derecha
        flexY3_der = int(results.pose_landmarks.landmark[16].y * height)
        flexY4_der = int(results.pose_landmarks.landmark[24].y * height)  # Punto 24: cadera derecha
        flexY5_der = int(results.pose_landmarks.landmark[26].y * height)  # Punto 26: rodilla derecha
        flexY6_der = int(results.pose_landmarks.landmark[28].y * height)  # Punto 28: tobillo derecho

        flexX1_izq = int(results.pose_landmarks.landmark[11].x * width)   # Punto 11: hombro izquierdo
        flexY1_izq = int(results.pose_landmarks.landmark[11].y * height)  
        flexX2_izq = int(results.pose_landmarks.landmark[13].x * width)   # Punto 13: codo izquierdo
        flexY2_izq = int(results.pose_landmarks.landmark[13].y * height)  
        flexX3_izq = int(results.pose_landmarks.landmark[15].x * width)   # Punto 15: muñeca izquierda
        flexY3_izq = int(results.pose_landmarks.landmark[15].y * height)
        flexY4_izq = int(results.pose_landmarks.landmark[23].y * height)  # Punto 23: cadera izquierda
        flexY5_izq = int(results.pose_landmarks.landmark[25].y * height)  # Punto 25: rodilla izquierda
        flexY6_izq = int(results.pose_landmarks.landmark[27].y * height)  # Punto 27: tobillo izquierdo

        # Punto medio entre las rodillas y los tobillos - eje Y
        flexYmed56_der = (flexY5_der + flexY6_der)/2
        flexYmed56_izq = (flexY5_izq + flexY6_izq)/2

        # Definir las localizaciones de los puntos de referencia
        flexPos1_der = np.array([flexX1_der, flexY1_der])
        flexPos2_der = np.array([flexX2_der, flexY2_der])
        flexPos3_der = np.array([flexX3_der, flexY3_der])

        flexPos1_izq = np.array([flexX1_izq, flexY1_izq])
        flexPos2_izq = np.array([flexX2_izq, flexY2_izq])
        flexPos3_izq = np.array([flexX3_izq, flexY3_izq])
        
        # Cálculo de los lados del triángulo
        flexLado1_der = np.linalg.norm(flexPos2_der - flexPos3_der)
        flexLado2_der = np.linalg.norm(flexPos1_der - flexPos3_der)
        flexLado3_der = np.linalg.norm(flexPos1_der - flexPos2_der)
        
        flexLado1_izq = np.linalg.norm(flexPos2_izq - flexPos3_izq)
        flexLado2_izq = np.linalg.norm(flexPos1_izq - flexPos3_izq)
        flexLado3_izq = np.linalg.norm(flexPos1_izq - flexPos2_izq)

        # Cálculo del ángulo
        flexAngulo_der = degrees(acos((flexLado1_der**2 + flexLado3_der**2 - flexLado2_der**2) / (2 * flexLado1_der * flexLado3_der)))
        flexAngulo_izq = degrees(acos((flexLado1_izq**2 + flexLado3_izq**2 - flexLado2_izq**2) / (2 * flexLado1_izq * flexLado3_izq)))

        # Condicionantes del ejercicio
        # Flag para verificar si la posición inicial de flexión es válida
        flexionValida = (
            flexY4_der < flexY3_der and flexY1_der < flexY3_der and flexYmed56_der < flexY3_der and
            flexY4_izq < flexY3_izq and flexY1_izq < flexY3_izq and flexYmed56_izq < flexY3_izq
        )

        # Evaluar el resto de los condicionantes solo si la posición inicial es válida
        #if flexionValida:
            # Verificar estado actual y actualizar banderas
        if flexAngulo_der >= 160 and flexAngulo_izq >= 160:
            flexionesArriba = True
        if flexionesArriba == True and 160 > flexAngulo_der > 90 and  160 > flexAngulo_izq > 90:
            flexionesMedioBajando = True
            flexionesArriba = False
        if flexionesMedioBajando == True and flexAngulo_der >= 160 and flexAngulo_izq >= 160:
            flexionesMedioBajando = False
            flexionesArriba = True
            flexionesContadorNoCompleta += 1
        if flexionesMedioBajando == True and flexAngulo_der <= 90 and flexAngulo_izq <= 90:
            flexionesMedioBajando = False
            flexionesAbajo = True
            flexionesBajadaCompleta = True
        if flexionesAbajo == True and 160 > flexAngulo_der > 90 and 160 > flexAngulo_izq > 90:
            flexionesAbajo  = False
            flexionesMedioSubiendo = True
        if flexionesMedioSubiendo == True and flexAngulo_der <= 90 and flexAngulo_izq <= 90:
            flexionesMedioSubiendo = False
            flexionesAbajo = True
            flexionesContadorNoCompleta += 1
        if flexionesMedioSubiendo == True and flexAngulo_der >= 160 and flexAngulo_izq >= 160:
            flexionesMedioSubiendo = False
            flexionesArriba = True
            flexionesSubidaCompleta = True
        if flexionesBajadaCompleta == True and flexionesSubidaCompleta == True:
            flexionesContador += 1
            flexionesBajadaCompleta = False
            flexionesSubidaCompleta = False
               
        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(flexAngulo_der), (flexX2_der, flexY2_der + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)
        cv2.putText(frame_rgb, "{:.2f}".format(flexAngulo_izq), (flexX2_izq, flexY2_izq + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (flexX1_der, flexY1_der), (flexX2_der, flexY2_der), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (flexX2_der, flexY2_der), (flexX3_der, flexY3_der), (0, 255, 0), 3)  # Codo a muñeca
        cv2.line(frame_rgb, (flexX1_izq, flexY1_izq), (flexX2_izq, flexY2_izq), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (flexX2_izq, flexY2_izq), (flexX3_izq, flexY3_izq), (0, 255, 0), 3)  # Codo a muñeca
        
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_flexiones_completas.text = str(flexionesContador)
        self.ids.etiq_contador_flexiones_nocompletas.text = str(flexionesContadorNoCompleta)

        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)



########################################################################################################################################################
# Sentadillas --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_sentadillas(self, frame_rgb, results, width, height):
        global sentadillasContador, sentadillasContadorNoCompleta, sentadillasArriba, sentadillasAbajo, sentadillasMedioBajando, sentadillasMedioSubiendo, sentadillasSubidaCompleta, sentadillasBajadaCompleta

        # Coordenadas        
        sentX1_der = int(results.pose_landmarks.landmark[24].x * width)   # Punto 24: cadera derecha
        sentY1_der = int(results.pose_landmarks.landmark[24].y * height)  
        sentX2_der = int(results.pose_landmarks.landmark[26].x * width)   # Punto 26: rodilla derecha
        sentY2_der = int(results.pose_landmarks.landmark[26].y * height)  
        sentX3_der = int(results.pose_landmarks.landmark[28].x * width)   # Punto 28: tobillo derecho
        sentY3_der = int(results.pose_landmarks.landmark[28].y * height)  

        sentX1_izq = int(results.pose_landmarks.landmark[23].x * width)   # Punto 23: cadera izquierda
        sentY1_izq = int(results.pose_landmarks.landmark[23].y * height)  
        sentX2_izq = int(results.pose_landmarks.landmark[25].x * width)   # Punto 25: rodilla izquierda
        sentY2_izq = int(results.pose_landmarks.landmark[25].y * height)  
        sentX3_izq = int(results.pose_landmarks.landmark[27].x * width)   # Punto 27: tobillo izquierdo
        sentY3_izq = int(results.pose_landmarks.landmark[27].y * height)  
        
        # Definir las localizaciones de los puntos de referencia
        sentPos1_der = np.array([sentX1_der, sentY1_der])
        sentPos2_der = np.array([sentX2_der, sentY2_der])
        sentPos3_der = np.array([sentX3_der, sentY3_der])

        sentPos1_izq = np.array([sentX1_izq, sentY1_izq])
        sentPos2_izq = np.array([sentX2_izq, sentY2_izq])
        sentPos3_izq = np.array([sentX3_izq, sentY3_izq])

        # Cálculo de los lados del triángulo
        sentLado1_der = np.linalg.norm(sentPos2_der - sentPos3_der)
        sentLado2_der = np.linalg.norm(sentPos1_der - sentPos3_der)
        sentLado3_der = np.linalg.norm(sentPos1_der - sentPos2_der)
        
        sentLado1_izq = np.linalg.norm(sentPos2_izq - sentPos3_izq)
        sentLado2_izq = np.linalg.norm(sentPos1_izq - sentPos3_izq)
        sentLado3_izq = np.linalg.norm(sentPos1_izq - sentPos2_izq)

        # Cálculo del ángulo
        sentAngulo_der = degrees(acos((sentLado1_der**2 + sentLado3_der**2 - sentLado2_der**2) / (2 * sentLado1_der * sentLado3_der)))
        sentAngulo_izq = degrees(acos((sentLado1_izq**2 + sentLado3_izq**2 - sentLado2_izq**2) / (2 * sentLado1_izq * sentLado3_izq)))

        # Condicionantes del ejercicio
        if sentAngulo_der >= 160 and sentAngulo_izq >= 160:
            sentadillasArriba = True
        if sentadillasArriba == True and 160 > sentAngulo_der > 90 and  160 > sentAngulo_izq > 90:
            sentadillasMedioBajando = True
            sentadillasArriba = False
        if sentadillasMedioBajando == True and sentAngulo_der >= 160 and sentAngulo_izq >= 160:
            sentadillasMedioBajando = False
            sentadillasArriba = True
            sentadillasContadorNoCompleta += 1
        if sentadillasMedioBajando == True and sentAngulo_der <= 90 and sentAngulo_izq <= 90:
            sentadillasMedioBajando = False
            sentadillasAbajo = True
            sentadillasBajadaCompleta = True
        if sentadillasAbajo == True and 160 > sentAngulo_der > 90 and 160 > sentAngulo_izq > 90:
            sentadillasAbajo  = False
            sentadillasMedioSubiendo = True
        if sentadillasMedioSubiendo == True and sentAngulo_der <= 90 and sentAngulo_izq <= 90:
            sentadillasMedioSubiendo = False
            sentadillasAbajo = True
            sentadillasContadorNoCompleta += 1
        if sentadillasMedioSubiendo == True and sentAngulo_der >= 160 and sentAngulo_izq >= 160:
            sentadillasMedioSubiendo = False
            sentadillasArriba = True
            sentadillasSubidaCompleta = True
        if sentadillasBajadaCompleta == True and sentadillasSubidaCompleta == True:
            sentadillasContador += 1
            sentadillasBajadaCompleta = False
            sentadillasSubidaCompleta = False

        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(sentAngulo_der), (sentX2_der, sentY2_der + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)
        cv2.putText(frame_rgb, "{:.2f}".format(sentAngulo_izq), (sentX2_izq, sentY2_izq + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (sentX1_der, sentY1_der), (sentX2_der, sentY2_der), (0, 255, 0), 3)  # Cadera a rodilla
        cv2.line(frame_rgb, (sentX2_der, sentY2_der), (sentX3_der, sentY3_der), (0, 255, 0), 3)  # Rodilla a tobillo
        cv2.line(frame_rgb, (sentX1_izq, sentY1_izq), (sentX2_izq, sentY2_izq), (0, 255, 0), 3)  # Cadera a rodilla
        cv2.line(frame_rgb, (sentX2_izq, sentY2_izq), (sentX3_izq, sentY3_izq), (0, 255, 0), 3)  # Rodilla a tobillo
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_sentadillas_completas.text = str(sentadillasContador)
        self.ids.etiq_contador_sentadillas_nocompletas.text = str(sentadillasContadorNoCompleta)
        
        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)


########################################################################################################################################################
# Zancada Atras --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_zancada_atras(self, frame_rgb, results, width, height):
        global zancadaContador, zancadaContadorIzq, zancadaContadorDer, abiertoDer, cerradoDer, abiertoIzq, cerradoIzq

        # Coordenadas        
        zatrasX1_der = int(results.pose_landmarks.landmark[24].x * width)   # Punto 24: cadera derecha
        zatrasY1_der = int(results.pose_landmarks.landmark[24].y * height)  
        zatrasX2_der = int(results.pose_landmarks.landmark[26].x * width)   # Punto 26: rodilla derecha
        zatrasY2_der = int(results.pose_landmarks.landmark[26].y * height)  
        zatrasX3_der = int(results.pose_landmarks.landmark[28].x * width)   # Punto 28: tobillo derecho
        zatrasY3_der = int(results.pose_landmarks.landmark[28].y * height)  

        zatrasX1_izq = int(results.pose_landmarks.landmark[23].x * width)   # Punto 23: cadera izquierda
        zatrasY1_izq = int(results.pose_landmarks.landmark[23].y * height)  
        zatrasX2_izq = int(results.pose_landmarks.landmark[25].x * width)   # Punto 25: rodilla izquierda
        zatrasY2_izq = int(results.pose_landmarks.landmark[25].y * height)  
        zatrasX3_izq = int(results.pose_landmarks.landmark[27].x * width)   # Punto 27: tobillo izquierdo
        zatrasY3_izq = int(results.pose_landmarks.landmark[27].y * height)  
        
        # Definir las localizaciones de los puntos de referencia
        zatrasPos1_der = np.array([zatrasX1_der, zatrasY1_der])
        zatrasPos2_der = np.array([zatrasX2_der, zatrasY2_der])
        zatrasPos3_der = np.array([zatrasX3_der, zatrasY3_der])

        zatrasPos1_izq = np.array([zatrasX1_izq, zatrasY1_izq])
        zatrasPos2_izq = np.array([zatrasX2_izq, zatrasY2_izq])
        zatrasPos3_izq = np.array([zatrasX3_izq, zatrasY3_izq])

        # Cálculo de los lados del triángulo
        zatrasLado1_der = np.linalg.norm(zatrasPos2_der - zatrasPos3_der)
        zatrasLado2_der = np.linalg.norm(zatrasPos1_der - zatrasPos3_der)
        zatrasLado3_der = np.linalg.norm(zatrasPos1_der - zatrasPos2_der)
        
        zatrasLado1_izq = np.linalg.norm(zatrasPos2_izq - zatrasPos3_izq)
        zatrasLado2_izq = np.linalg.norm(zatrasPos1_izq - zatrasPos3_izq)
        zatrasLado3_izq = np.linalg.norm(zatrasPos1_izq - zatrasPos2_izq)

        # Cálculo del ángulo
        zatrasAngulo_der = degrees(acos((zatrasLado1_der**2 + zatrasLado3_der**2 - zatrasLado2_der**2) / (2 * zatrasLado1_der * zatrasLado3_der)))
        zatrasAngulo_izq = degrees(acos((zatrasLado1_izq**2 + zatrasLado3_izq**2 - zatrasLado2_izq**2) / (2 * zatrasLado1_izq * zatrasLado3_izq)))

        # Condicionantes del ejercicio
        if zatrasAngulo_der >= 150:
            abiertoDer = True
        if abiertoDer == True and cerradoDer == False and zatrasAngulo_der <= 100 and zatrasY2_izq > zatrasY2_der:
            cerradoDer = True
        if abiertoDer == True and cerradoDer == True and zatrasAngulo_der >= 150 :
            zancadaContadorDer += 1
            abiertoDer = False
            cerradoDer = False

        if zatrasAngulo_izq >= 150:
            abiertoIzq = True
        if abiertoIzq == True and cerradoIzq == False and zatrasAngulo_izq <= 100 and zatrasY2_der > zatrasY2_izq:
            cerradoIzq = True
        if abiertoIzq == True and cerradoIzq == True and zatrasAngulo_izq >= 150 :
            zancadaContadorIzq += 1
            abiertoIzq = False
            cerradoIzq = False
            
        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(zatrasAngulo_der), (zatrasX2_der, zatrasY2_der - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)
        cv2.putText(frame_rgb, "{:.2f}".format(zatrasAngulo_izq), (zatrasX2_izq, zatrasY2_izq - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (zatrasX1_der, zatrasY1_der), (zatrasX2_der, zatrasY2_der), (0, 255, 0), 3)  # Cadera a rodilla
        cv2.line(frame_rgb, (zatrasX2_der, zatrasY2_der), (zatrasX3_der, zatrasY3_der), (0, 255, 0), 3)  # Rodilla a tobillo
        cv2.line(frame_rgb, (zatrasX1_izq, zatrasY1_izq), (zatrasX2_izq, zatrasY2_izq), (0, 255, 0), 3)  # Cadera a rodilla
        cv2.line(frame_rgb, (zatrasX2_izq, zatrasY2_izq), (zatrasX3_izq, zatrasY3_izq), (0, 255, 0), 3)  # Rodilla a tobillo
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_zatras_der.text = str(zancadaContadorDer)
        self.ids.etiq_contador_zatras_izq.text = str(zancadaContadorIzq)
        self.ids.etiq_contador_zatras_tot.text = str(zancadaContadorDer + zancadaContadorIzq)
        
        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)


########################################################################################################################################################
# Crunch Abdominal --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_crunch_abdominal(self, frame_rgb, results, width, height):
        global abdominalesContador, abdominalesEstirado, abdominalesContraido

        # Coordenadas de los brazos
        abdX1_der = int(results.pose_landmarks.landmark[12].x * width)   # Punto 12: hombro derecho
        abdY1_der = int(results.pose_landmarks.landmark[12].y * height)
        abdX2_der = int(results.pose_landmarks.landmark[24].x * width)   # Punto 24: cadera derecha
        abdY2_der = int(results.pose_landmarks.landmark[24].y * height)         
        abdX3_der = int(results.pose_landmarks.landmark[30].x * width)   # Punto 30: talon derecho
        abdY3_der = int(results.pose_landmarks.landmark[30].y * height)

        # Definir las localizaciones de los puntos de referencia
        abdPos1_der = np.array([abdX1_der, abdY1_der])
        abdPos2_der = np.array([abdX2_der, abdY2_der])
        abdPos3_der = np.array([abdX3_der, abdY3_der])

        # Cálculo de los lados del triángulo
        abdLado1_der = np.linalg.norm(abdPos2_der - abdPos3_der)
        abdLado2_der = np.linalg.norm(abdPos1_der - abdPos3_der)
        abdLado3_der = np.linalg.norm(abdPos1_der - abdPos2_der)

        # Cálculo del ángulo
        abdAngulo_der = degrees(acos((abdLado1_der**2 + abdLado3_der**2 - abdLado2_der**2) / (2 * abdLado1_der * abdLado3_der)))

        # Condicionantes del ejercicio
        if abdAngulo_der >= 165:
            abdominalesEstirado = True
        if abdominalesEstirado == True and abdominalesContraido == False and abdAngulo_der <= 140:
            abdominalesContraido = True
        if abdominalesEstirado == True and abdominalesContraido == True and abdAngulo_der >= 165:
            abdominalesContador += 1
            abdominalesEstirado = False
            abdominalesContraido = False

        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(abdAngulo_der), (abdX2_der, abdY2_der - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (abdX1_der, abdY1_der), (abdX2_der, abdY2_der), (0, 255, 0), 3)  # Hombro a cadera
        cv2.line(frame_rgb, (abdX2_der, abdY2_der), (abdX3_der, abdY3_der), (0, 255, 0), 3)  # Cadera a talon
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_abdominales_der.text = str(abdominalesContador)
        
        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

              
        #pass

########################################################################################################################################################
# Dominadas --------------------------------------------------------------------------------------------------------------------------------------------
    def contar_dominadas(self, frame_rgb, results, width, height):
        global dominadasContador, dominadasContadorNoCompleta, dominadasArriba, dominadasAbajo, dominadasMedioBajando, dominadasMedioSubiendo, dominadasSubidaCompleta, dominadasBajadaCompleta

        # Coordenadas de los brazos
        domX1_der = int(results.pose_landmarks.landmark[12].x * width)   # Punto 12: hombro derecho
        domY1_der = int(results.pose_landmarks.landmark[12].y * height)  
        domX2_der = int(results.pose_landmarks.landmark[14].x * width)   # Punto 14: codo derecho
        domY2_der = int(results.pose_landmarks.landmark[14].y * height)  
        domX3_der = int(results.pose_landmarks.landmark[16].x * width)   # Punto 16: muñeca derecha
        domY3_der = int(results.pose_landmarks.landmark[16].y * height)  

        domX1_izq = int(results.pose_landmarks.landmark[11].x * width)   # Punto 11: hombro izquierdo
        domY1_izq = int(results.pose_landmarks.landmark[11].y * height)  
        domX2_izq = int(results.pose_landmarks.landmark[13].x * width)   # Punto 13: codo izquierdo
        domY2_izq = int(results.pose_landmarks.landmark[13].y * height)  
        domX3_izq = int(results.pose_landmarks.landmark[15].x * width)   # Punto 15: muñeca izquierda
        domY3_izq = int(results.pose_landmarks.landmark[15].y * height)  

        # Definir las localizaciones de los puntos de referencia
        domPos1_der = np.array([domX1_der, domY1_der])
        domPos2_der = np.array([domX2_der, domY2_der])
        domPos3_der = np.array([domX3_der, domY3_der])

        domPos1_izq = np.array([domX1_izq, domY1_izq])
        domPos2_izq = np.array([domX2_izq, domY2_izq])
        domPos3_izq = np.array([domX3_izq, domY3_izq])
        
        # Cálculo de los lados del triángulo
        domLado1_der = np.linalg.norm(domPos2_der - domPos3_der)
        domLado2_der = np.linalg.norm(domPos1_der - domPos3_der)
        domLado3_der = np.linalg.norm(domPos1_der - domPos2_der)
        
        domLado1_izq = np.linalg.norm(domPos2_izq - domPos3_izq)
        domLado2_izq = np.linalg.norm(domPos1_izq - domPos3_izq)
        domLado3_izq = np.linalg.norm(domPos1_izq - domPos2_izq)

        # Cálculo del ángulo
        domAngulo_der = degrees(acos((domLado1_der**2 + domLado3_der**2 - domLado2_der**2) / (2 * domLado1_der * domLado3_der)))
        domAngulo_izq = degrees(acos((domLado1_izq**2 + domLado3_izq**2 - domLado2_izq**2) / (2 * domLado1_izq * domLado3_izq)))

        # Condicionantes del ejercicio de dominadas
        if domAngulo_der >= 160 and domAngulo_izq >= 160:
            dominadasAbajo = True
        if dominadasAbajo == True and 160 > domAngulo_der > 100 and 160 > domAngulo_izq > 100:
            dominadasMedioSubiendo = True
            dominadasAbajo = False
        if dominadasMedioSubiendo == True and domAngulo_der >= 160 and domAngulo_izq >= 160:
            dominadasMedioSubiendo = False
            dominadasAbajo = True
            dominadasContadorNoCompleta += 1
        if dominadasMedioSubiendo == True and domAngulo_der <= 100 and domAngulo_izq <= 100:
            dominadasMedioSubiendo = False
            dominadasArriba = True
            dominadasSubidaCompleta = True
        if dominadasArriba == True and 160 > domAngulo_der > 100 and 160 > domAngulo_izq > 100:
            dominadasArriba = False
            dominadasMedioBajando = True
        if dominadasMedioBajando == True and domAngulo_der <= 100 and domAngulo_izq <= 100:
            dominadasMedioBajando = False
            dominadasArriba = True
            dominadasContadorNoCompleta += 1
        if dominadasMedioBajando == True and domAngulo_der >= 160 and domAngulo_izq >= 160:
            dominadasMedioBajando = False
            dominadasAbajo = True
            dominadasBajadaCompleta = True
        if dominadasSubidaCompleta == True and dominadasBajadaCompleta == True:
            dominadasContador += 1
            dominadasSubidaCompleta = False
            dominadasBajadaCompleta = False

        # Mostrar los ángulos en el frame
        cv2.putText(frame_rgb, "{:.2f}".format(domAngulo_der), (domX2_der, domY2_der + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)
        cv2.putText(frame_rgb, "{:.2f}".format(domAngulo_izq), (domX2_izq, domY2_izq + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3, lineType=cv2.LINE_AA, bottomLeftOrigin=True)

        # Dibujar líneas entre los puntos de los brazos
        cv2.line(frame_rgb, (domX1_der, domY1_der), (domX2_der, domY2_der), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (domX2_der, domY2_der), (domX3_der, domY3_der), (0, 255, 0), 3)  # Codo a muñeca
        cv2.line(frame_rgb, (domX1_izq, domY1_izq), (domX2_izq, domY2_izq), (0, 255, 0), 3)  # Hombro a codo
        cv2.line(frame_rgb, (domX2_izq, domY2_izq), (domX3_izq, domY3_izq), (0, 255, 0), 3)  # Codo a muñeca
        
        # Actualizar etiquetas de contadores
        self.ids.etiq_contador_dominadas_completas.text = str(dominadasContador)
        self.ids.etiq_contador_dominadas_nocompletas.text = str(dominadasContadorNoCompleta)

        
        #Mostrar ejercicio seleccionado
        cv2.putText(frame_rgb, f"Ejercicio: {ejercicio_seleccionado}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA, bottomLeftOrigin=True)



########################################################################################################################################################
# Camara y visualizacion --------------------------------------------------------------------------------------------------------------------------------------------
    def visualizar(self, dt):
        global cap, ejercicio_seleccionado
        global bicepsContadorTot, bicepsContadorDer, bicepsContadorIzq, bicepsAbiertoDer, bicepsCerradoDer, bicepsAbiertoIzq, bicepsCerradoIzq
        global flexionesContador, flexionesContadorNoCompleta, flexionesArriba, flexionesAbajo, flexionesMedioBajando, flexionesMedioSubiendo, flexionesSubidaCompleta, flexionesBajadaCompleta
        global sentadillasContador, sentadillasContadorNoCompleta, sentadillasArriba, sentadillasAbajo, sentadillasMedioBajando, sentadillasMedioSubiendo, sentadillasSubidaCompleta, sentadillasBajadaCompleta
        global zancadaContador, zancadaContadorIzq, zancadaContadorDer, abiertoDer, cerradoDer, abiertoIzq, cerradoIzq
        global abdominalesContador, abdominalesEstirado, abdominalesContraido
        global dominadasContador, dominadasContadorNoCompleta, dominadasArriba, dominadasAbajo, dominadasMedioBajando, dominadasMedioSubiendo, dominadasSubidaCompleta, dominadasBajadaCompleta
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, -1)

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            height, width, _ = frame.shape

            if results.pose_landmarks is not None:
                if ejercicio_seleccionado == "Biceps":
                    self.contar_biceps(frame_rgb, results, width, height)
                elif ejercicio_seleccionado == "Flexiones":
                    self.contar_flexiones(frame_rgb, results, width, height)
                elif ejercicio_seleccionado == "Sentadillas":
                    self.contar_sentadillas(frame_rgb, results, width, height)
                elif ejercicio_seleccionado == "Zancada atras":
                    self.contar_zancada_atras(frame_rgb, results, width, height)
                elif ejercicio_seleccionado == "Crunch abdominal":
                    self.contar_crunch_abdominal(frame_rgb, results, width, height)
                elif ejercicio_seleccionado == "Dominadas":
                    self.contar_dominadas(frame_rgb, results, width, height)

            # Mostrar la imagen en la interfaz
            buf = frame_rgb.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            # Actualiza la textura del Image en el layout
            self.ids.camera_feed.texture = texture
  
    def on_stop(self):
        # Liberar la cámara al cerrar la app
        cap.release()

########################################################################################################################################################
# Main --------------------------------------------------------------------------------------------------------------------------------------------

class main(App):
    def build(self):
        self.title = "APK ESCRITORIO LBM"
        self.layout = MainLayout()

        return self.layout


if __name__ == '__main__':
    main().run()
