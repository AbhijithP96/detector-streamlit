import cv2

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('head.avi', cv2.VideoWriter.fourcc(*'XVID'), fps=30, frameSize=(640,480), isColor=True)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    writer.write(frame)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()