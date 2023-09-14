import cv2
import numpy as np
import imutils

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Load YOLO weights and configuration
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes for vehicles (modify this based on your YOLO configuration)
vehicle_classes = ["car", "truck", "bus", "motorcycle"]  # Add other vehicle classes as needed

# Initialize video capture for the two videos
cap1 = cv2.VideoCapture(r"C:\Users\chapp\Downloads\Untitled video - Made with Clipchamp.mp4")  # Replace with the path to your first video
cap2 = cv2.VideoCapture(r"C:\Users\chapp\Downloads\Untitled video - Made with Clipchamp (1).mp4")  # Replace with the path to your second video

while True:
    ret1, frame1 = cap1.read()  # Read a frame from the first video
    ret2, frame2 = cap2.read()  # Read a frame from the second video

    if not ret1 or not ret2:
        break

    # Perform object detection on the frames
    blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob1)
    outs1 = net.forward(output_layers)

    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)


    def count_vehicles(outs, classes):
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id < len(classes) and confidence > 0.5:
                    if classes[class_id] in vehicle_classes:
                        class_ids.append(class_id)
        return len(class_ids)


    vehicle_count1 = count_vehicles(outs1, vehicle_classes)
    vehicle_count2 = count_vehicles(outs2, vehicle_classes)

    # Display the vehicle count on the frames
    cv2.putText(frame1, f"Vehicles: {vehicle_count1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame2, f"Vehicles: {vehicle_count2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame1 = imutils.resize(frame1, width=600)
    frame2 = imutils.resize(frame2, width=600)
    # Display the frames
    cv2.imshow("Video 1", frame1)
    cv2.imshow("Video 2", frame2)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF == ord('q')

    if vehicle_count1 <= vehicle_count2:
        green_overlay = np.zeros_like(frame2)
        green_overlay[:] = (0, 255, 0)
        alpha = 0.5
        frame2= cv2.addWeighted(frame2, 1 - alpha, green_overlay, alpha, 0)
        cv2.imshow("Video 3", frame2)
        red_overlay = np.zeros_like(frame1)
        red_overlay[:] = (0, 0, 255)
        frame1 = cv2.addWeighted(frame1, 1 - alpha, red_overlay, alpha, 0)
        cv2.imshow("Video 4", frame1)

    if vehicle_count1 > vehicle_count2:
        green_overlay = np.zeros_like(frame1)
        green_overlay[:] = (0, 255, 0)
        alpha = 0.5
        frame1= cv2.addWeighted(frame1, 1 - alpha, green_overlay, alpha, 0)
        cv2.imshow("Video 3", frame1)
        red_overlay = np.zeros_like(frame2)
        red_overlay[:] = (0, 0, 255)
        frame2 = cv2.addWeighted(frame2, 1 - alpha, red_overlay, alpha, 0)
        cv2.imshow("Video 4", frame2)
# Release the video captures and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
