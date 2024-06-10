import cv2
import numpy as np
yolo=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes=[]
with open("coco.names","r") as file:
    classes=[line.strip() for line in file.readlines() ]
layer_names= yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed=(0,0,255)
coloYellow=(0, 255, 255) #changed

#loading images
name="image.jpg"
img=cv2.imread(name)
height, width, channels = img.shape

#detecting object
blob=cv2.dnn.blobFromImage(img, 0.00392, (416, 416),(0, 0, 0),True, crop=False) 

yolo.setInput(blob)

outputs = yolo.forward(output_layers)

class_ids=[]
confidences=[] #removes noise
boxes=[]    #defining all detected objects
for output in outputs:
    for detection in output:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence = scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w= int(detection[2]*width)
            h=int(detection[3]*height)

            x=int(center_x - w/2)
            y=int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#NMS is non maxima supression
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(len(boxes))
print(len(indexes))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label =str(classes[class_ids[i]])
        cv2.rectangle(img,(x,y),(x + w, y + h),coloYellow,3)
        cv2.putText(img,label,(x,y+10),cv2.FONT_HERSHEY_PLAIN,8,colorRed,8)

cv2.imwrite("output.jpg",img)



