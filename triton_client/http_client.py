import cv2
import tritonclient.http as httpclient
import argparse
from utils import *
import time
SAVE_INTERMEDIATE_IMAGES = False

def parse_argument():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url', type=str,default="localhost:8000")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source',type=str,required=True)
    parser.add_argument('--video_save_path',type=str,default="",required=False)
    opt=parser.parse_args()
    return opt

if __name__ == "__main__":
    opt=parse_argument()
    # Setting up client
    client = httpclient.InferenceServerClient(url=opt.url)
    model_name=None
    labels={}
    if opt.model=="yolov5":
        model_name="yolov5_trt"
        input_name="images"
        input_shape=(640,640)
        output_name="output0"
        with open("labels/yolo_coco_labels.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                id,clx=line.split(":")
                labels[int(id)]=clx[1:-1]
    else:
        raise BaseException("Not supported model")
    source=opt.source
    type="stream"
    if not source.isnumeric() and source.split('.')[-1] in {"jpg","png","jpeg"}:
        type="image"
    cap=cv2.VideoCapture(source if not source.isnumeric() else int(source))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if opt.video_save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter(opt.video_save_path,fourcc,fps,(width,height))
    while cap.isOpened():
        success,frame=cap.read()
        if not success:
            break
        start_time=time.time()
        preprocessed_image,rs_image = input_preprocess(frame,input_shape)
        detection_input = httpclient.InferInput(
            "images", preprocessed_image.shape, datatype="FP32")
        detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)
        # Query the server
        detection_response = client.infer(
            model_name=model_name, inputs=[detection_input])
        frame=postprocess(rs_image,detection_response.as_numpy("output0"),labels,opt.model)
        resize_size=max(width,height)
        frame = cv2.resize(frame, (resize_size, resize_size))
        if type=="stream":
            cv2.imshow("Detection",frame)
        if opt.video_save_path:
            video.write(frame)
        if type=="image":
            cv2.imwrite("result.png",frame)
            print("Saved to result.png")
        # print(time.time()-start_time)
        wait_time=int(1000/fps-(time.time()-start_time)*1000)
        if cv2.waitKey(wait_time if wait_time>0 else 1) & 0xFF==ord('q'):
            break
    cap.release()
    if opt.video_save_path:
        video.release()
        print("Saved result to ",opt.video_save_path)
    cv2.destroyAllWindows()

