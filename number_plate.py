import cv2

haarcascades = "model\haarcascade_russian_plate_number.xml"

# RTSP URL
rtsp_url = 'rtsp://admin:Admin123@172.168.9.8:554/Streaming/Channels/1'

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# # Check if the stream is opened successfully
# if not cap.isOpened():
#     print("Error: Cannot open the RTSP stream")
#     exit()

# # Read and display the frames from the stream
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Display the frame
#     cv2.imshow('RTSP Stream', frame)

#     # Press 'q' to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close the window
# cap.release()
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)

# cap.set(3, 640)
# cap.set(4, 480)

min_area=500
count=0

while True:
    success, img = cap.read()
    
    plate_cascade=cv2.CascadeClassifier(haarcascades)
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates=plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w*h

        if area > min_area:
            cv2.rectangle(img, (x,y), (x*y, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,0,255),2)

            img_roi=img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scanned_img_"+ str(count)+".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img, "Plate Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
        cv2.imshow("Results",img)
        cv2.waitKey(500)
        count += 1