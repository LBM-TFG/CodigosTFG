import cv2

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
  ret, imagen = cap.read()
  #imagen = cv2.flip(imagen, 1)
  if ret == True:
    cv2.imshow('video', imagen)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
  else: break
  
captura.release()
cv2.destroyAllWindows()
