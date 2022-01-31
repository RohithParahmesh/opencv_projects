import cv2



video = cv2.VideoCapture('dashcam_footage2.mp4')


car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

   
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    print(pedestrian)
    


    
    for (x,y,w,h) in cars:  
        cv2.rectangle(frame,(x, y),(x+w,y+h),(0, 0, 255),2)
        if w>80 or h>80:
            cv2.putText(frame, "Car,Warning!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0) , 2)

    for (x,y,w,h) in pedestrian:  
        cv2.rectangle(frame,(x, y),(x+w,y+h),( 255 , 0, 0 ),2)
        if w>80 or h>80:
            cv2.putText(frame, "Pedestrian,Warning!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0) , 2)

    


    

    
    cv2.imshow('car detector', frame)

    
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()
    

print("end")