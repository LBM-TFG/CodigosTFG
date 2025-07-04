import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Lee el vídeo
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
     static_image_mode=False) as pose:

     while True:
          ret, frame = cap.read()
          if ret == False:
               break
          
          # Recoge dimensiones del vídeo y pasa de BGR a RGB
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = pose.process(frame_rgb)

          # Añadir marcas a la posición del cuerpo
          if results.pose_landmarks is not None:
           mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))            

          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == 27:
               break

cap.release()
cv2.destroyAllWindows()
