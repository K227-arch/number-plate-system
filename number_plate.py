import cv2
import subprocess
import numpy as np
import threading

# Path to the Haar Cascade XML file
haarcascades = "model/haarcascade_russian_plate_number.xml"

# IP address
ip_address = "172.168.9.8"
rtsp_url = f"rtsp://admin:Admin123@{ip_address}:554/Streaming/Channels/1"

# FFmpeg command to stream video
ffmpeg_command = [
    "ffmpeg",
    "-i", rtsp_url,
    "-fflags", "flush_packets",
    "-max_delay", "0.01",
    "-flags", "-global_header",
    "-hls_time", "0.5",
    "-hls_list_size", "1",
    "-vcodec", "copy",
    "-y", "./index.m3u8"
]

# Function to run the FFmpeg command
def run_ffmpeg():
    try:
        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print("FFmpeg command executed successfully")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing FFmpeg command: {e}")
        print(f"Output: {e.output}")

# Start the FFmpeg command in a separate thread
ffmpeg_thread = threading.Thread(target=run_ffmpeg)
ffmpeg_thread.start()

# Minimum area of the detected plate to be considered valid
min_area = 500
count = 0

# OpenCV processing part
plate_cascade = cv2.CascadeClassifier(haarcascades)
cap = cv2.VideoCapture(rtsp_url)

while True:
    success, image = cap.read()
    if not success:
        break

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

# Close OpenCV windows and release resources
cap.release()
cv2.destroyAllWindows()

# Wait for the FFmpeg thread to finish
ffmpeg_thread.join()
