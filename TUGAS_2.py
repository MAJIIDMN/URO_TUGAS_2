from ultralytics import YOLO
import cv2
import math 
# start video
cap = cv2.VideoCapture('object_video_manual.mp4')

# model(YOLO)
model = YOLO("yolo-Weights/yolov8n.pt")

while True:
    #read Video
    success, img = cap.read()
    #Resize video (ukuranya dipas in)
    imS = cv2.resize(img, (960, 540))
    #pake YOLO buat track object
    results = model.track(imS, stream=True)
    #proses per Frame
    for r in results:
        #Load nama-nama object
        classes_names = r.names
        for box in r.boxes:
           if box.conf[0] > 0.4:
               x1, y1, x2, y2 = box.xyxy[0]
               x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

               # bikin kotak diluar benda
               cv2.rectangle(imS, (x1, y1), (x2, y2), (255, 0, 255), 3)

               # kecocokan
               confidence = math.ceil((box.conf[0]*100))/100
               print("Confidence --->",confidence)

               # Nama benda
               cls = int(box.cls[0])
               print("Class name -->", classes_names[cls])

               # detail object
               org = [x1, y1]
               font = cv2.FONT_HERSHEY_SIMPLEX
               fontScale = 0.7
               color = (255, 0, 0)
               thickness = 2
               name = classes_names[cls] +' '+ str(confidence)

               cv2.putText(imS, name, org, font, fontScale, color, thickness)
    #tampilkan hasil
    cv2.imshow('Object_Track', imS)
    #tekan 'q' untuk stop, atau tunggu video habis
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()