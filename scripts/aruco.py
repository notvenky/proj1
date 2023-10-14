import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(6, cv2.CAP_V4L2)  # Use your camera index instead of 6
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Define the ArUco dictionary and parameters
dict_type = aruco.DICT_4X4_50
aruco_dict = aruco.getPredefinedDictionary(dict_type)
# see https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
parameters = aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
# import ipdb; ipdb.set_trace()

detector = aruco.ArucoDetector(aruco_dict, parameters)
# Dimensions of the marker in inches
marker_length = 1.0  

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the image to grayscale
    center = frame.shape
    # w = 640
    # h = 480
    # x = center[1]/2 - w/2
    # y = center[0]/2 - h/2
    # frame = frame[int(y):int(y+h), int(x):int(x+w)]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    alpha = 1.
    beta = 0.0
    gray = alpha * gray + beta
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    # Detect the markers in the image
    # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    # import ipdb; ipdb.set_trace()
    # If some markers are found, process them
    if ids is not None:
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        
        # for i in range(ids.size):
        #     # If you have camera calibration data, you can uncomment the following line to estimate the pose of the markers.
        #     # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_length, mtx, dist)
        #     cv2.putText(frame_markers, "ID: "+str(ids[i]), tuple(corners[i][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
    else:
        frame_markers = frame

    # frame_markers = aruco.drawDetectedMarkers(frame_markers, rejectedImgPoints, np.zeros(len(rejectedImgPoints), dtype=int), borderColor=(125, 125, 0))
    
    # Display the resulting frame
    cv2.imshow('Frame Markers', frame_markers)
    
    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()