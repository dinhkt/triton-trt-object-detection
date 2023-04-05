import cv2
import numpy as np


def postprocess(img,output,labels,model):
    if model=="yolov5":
        return yolo_postprocess(img,output,labels)
    else:
        print("not supported")

def NMS(boxes, confThresh=0.25, overlapThresh=0.45):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")               
    # pick boxes have confidence > threshold
    boxes=boxes[boxes[...,4]>confThresh]
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]-boxes[:,2]/2
    y1 = boxes[:,1]-boxes[:,3]/2
    x2 = boxes[:,0]+boxes[:,2]/2
    y2 = boxes[:,1]+boxes[:,3]/2
    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
    	# grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the integer data type
    return boxes[pick]


def input_preprocess(raw_bgr_image, input_shape=(640,640)):
    input_w, input_h = input_shape
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    image = cv2.resize(image, (tw, th))
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    blob = image.astype(np.float32)
    blob /= 255.0
    blob = np.transpose(blob, [2, 0, 1])
    return np.expand_dims(blob,0),image

def yolo_postprocess(img,outputs,lab):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for ig in outputs:
        ig=NMS(ig)
        for pred in ig:
            label=np.argmax(pred[5:])
            cfd_score=round(pred[4],2)
            pred=pred.astype(int)
            x1,y1,x2,y2=pred[0]-pred[2]//2,pred[1]-pred[3]//2,pred[0]+pred[2]//2,pred[1]+pred[3]//2
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            img = cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (0,255,0), -1)
            cv2.putText(img, lab[label]+" "+ str(cfd_score), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,26,12), 2)
    return img
