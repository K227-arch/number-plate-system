import cv2

# Path to the Haar Cascade XML file
haarcascades = "model/haarcascade_russian_plate_number.xml"

# RTSP URL
rtsp_url = 'rtsp://admin:Admin123@172.168.9.8:554/Streaming/Channels/1'

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# Check if the video capture object was initialized successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Minimum area of the detected plate to be considered valid
min_area = 500
count = 0

while True:
    success, img = cap.read()
    
    # Check if frame was read successfully
    if not success:
        print("Failed to read frame from video stream")
        break
    
    # Load the Haar Cascade classifier for detecting license plates
    plate_cascade = cv2.CascadeClassifier(haarcascades)
    
    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect plates in the grayscale frame
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x, y, w, h) in plates:
        area = w * h
        
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Extract the region of interest (ROI) containing the detected plate
            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)
    
    # Display the resulting frame
    cv2.imshow("Result", img)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the image of the detected plate when 's' key is pressed
        cv2.imwrite("plates/scanned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()