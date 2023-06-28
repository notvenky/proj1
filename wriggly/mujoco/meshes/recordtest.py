import cv2

cap = cv2.VideoCapture(0) # Change the number if your camera is not the default one

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') # You can use others like 'MJPG', 'X264' etc.
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) # Change the resolution if needed

while(cap.isOpened()):
    ret, frame = cap.read() # Read the frame from the webcam
    if ret==True:
        out.write(frame) # Write the frame into the file 'output.avi'

        cv2.imshow('frame',frame) # Show the frame on the screen
        if cv2.waitKey(1) & 0xFF == ord('q'): # Exit if 'q' is pressed
            break
    else:
        break

# Release everything after recording
cap.release()
out.release()
cv2.destroyAllWindows()