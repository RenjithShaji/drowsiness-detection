import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Face Detection and Eye Detection components
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Minimum eye aspect ratio for drowsiness detection
min_eye_aspect_ratio = 0.25

# Open the video file 
cap = cv2.VideoCapture(r'C:\Users\User\OneDrive\Desktop\project\Demo.mp4')

# Initialize text-to-speech engine
engine = pyttsx3.init()

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

               
                

                # Calculate eye aspect ratio
                eye_aspect_ratio = (w + h) / (2 * iw)

                # Check for drowsiness
                if eye_aspect_ratio < min_eye_aspect_ratio:
                    cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Alert in voice
                    engine.say("Warning! Drowsiness Detected.")
                    engine.runAndWait()

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()