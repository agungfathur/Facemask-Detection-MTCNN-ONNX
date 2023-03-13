import cv2
from mtcnn_cv2.mtcnn_opencv import MTCNN
from classifier.mask_classifier import classifier
import numpy as np
import time

detector = MTCNN()

input = 0
# input = "file2.mp4"
cap = cv2.VideoCapture(input)

frame_width = 480
frame_height = 320
size = (frame_width, frame_height)

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    start_time = time.time() # start time of the loop

    if ret == True:
        image = cv2.resize(frame, size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)
        
        if len(result) > 0:
            for face in result:                                
                bounding_box = face["box"]
                keypoints = face['keypoints']

                x = bounding_box[0]
                y = bounding_box[1]
                w = bounding_box[2]
                h = bounding_box[3]

                crop_img = image[y:y+h, x:x+w]
                classes = classifier(crop_img)

                #to show bounding box on faces
                cv2.rectangle(image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                            (0,155,255),
                            2)
                  
                # # to show keypoint on faces
                # cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
                # cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
                # cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
                # cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
                # cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

                # classify using mask, without mask, and incorect mask
                cv2.putText(image, classes, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,155,255), 2)
     
        else:
            pass    

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(image, "FPS :" + (str(round(fps,2))), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Frame', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows