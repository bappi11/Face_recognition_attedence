import cv2

v_name = f"./video/171-115-199.MOV"
cap = cv2.VideoCapture(v_name)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
i = 1

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        roi_color = frame[y:y + h, x:x + w]
        name = f"./images/171-115-199/{i}.jpg"
        cv2.imwrite(name, roi_color)

    cv2.imshow('Video', frame)
    i += 1
    k = cv2.waitKey(1) or i == 1000
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
