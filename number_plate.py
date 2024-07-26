import cv2
import subprocess
import numpy as np

# Path to the Haar Cascade XML file
haarcascades = "model/haarcascade_russian_plate_number.xml"

# RTSP URL
rtsp_url = 'rtsp://admin:Admin123@172.168.9.8:554/Streaming/Channels/1'

# FFmpeg command to capture frames from RTSP stream
ffmpeg_command = [
    'ffmpeg',
    '-i', rtsp_url,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo', '-'
]

# Minimum area of the detected plate to be considered valid
min_area = 500
count = 0

# Open a subprocess to run the FFmpeg command
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=10**8)

# Load the Haar Cascade classifier for detecting license plates
plate_cascade = cv2.CascadeClassifier(haarcascades)

while True:
    # Read the next frame from the FFmpeg process
    raw_image = process.stdout.read(640 * 480 * 3)  # Width * Height * 3 (for RGB)
    if len(raw_image) == 0:
        break
    
    # Convert the raw bytes to a numpy array
    image = np.frombuffer(raw_image, dtype=np.uint8).reshape((480, 640, 3))
    
    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect plates in the grayscale frame
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    for (x, y, w, h) in plates:
        area = w * h
        
        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            # Extract the region of interest (ROI) containing the detected plate
            img_roi = image[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)
    
    # Display the resulting frame
    cv2.imshow("Result", image)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the image of the detected plate when 's' key is pressed
        cv2.imwrite("plates/scanned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(image, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", image)
        cv2.waitKey(500)
        count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the FFmpeg process
process.stdout.close()
process.wait()

# Release resources and close OpenCV windows
cv2.destroyAllWindows()