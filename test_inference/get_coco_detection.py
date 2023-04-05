
import json
import os

ANNO_FILE="/home/ncl/ktd/datasets/coco/anno/instances_val2017.json"
ANNO_FD="../mAP/input/ground-truth/"
  
id_to_name={}
from collections import defaultdict
bboxs=defaultdict(list)
detection_labels={}
with open("../triton_client/labels/yolo_coco_labels.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        id,clx=line.split(":")
        detection_labels[clx[1:-1]]=int(id)
print(detection_labels)
cat={}
with open(ANNO_FILE,"r") as f:
    data=json.load(f)
    for ctg in data["categories"]:
        cat[ctg["id"]]=ctg["name"]
    for img in data["images"]:
        id_to_name[img["id"]]=(img["file_name"],img["width"],img["height"])
    print(len(data["annotations"]),len(data["images"]))
    for anno in data["annotations"]:
        bbox=anno["bbox"]
        if cat[anno["category_id"]] in detection_labels:
            lab=detection_labels[cat[anno["category_id"]]]
        else:
            continue
        bboxs[anno["image_id"]].append(lab)
        bboxs[anno["image_id"]].append(int(bbox[0]))
        bboxs[anno["image_id"]].append(int(bbox[1]))
        bboxs[anno["image_id"]].append(int(bbox[0]+bbox[2]))
        bboxs[anno["image_id"]].append(int(bbox[1]+bbox[3]))
    for id in id_to_name:
        txt_file_name=id_to_name[id][0].split(".")[0]+".txt"
        txt_file_path=os.path.join(ANNO_FD,txt_file_name)
        fw=open(txt_file_path,"w")
        idx=0
        while idx<len(bboxs[id]):
            fw.write(str(bboxs[id][idx])+" "+str(bboxs[id][idx+1])+" "+str(bboxs[id][idx+2])+" "+str(bboxs[id][idx+3])+" "+str(bboxs[id][idx+4])+"\n")
            idx+=5
        fw.close()
