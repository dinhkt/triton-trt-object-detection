import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc,InferResult
import cv2 
import argparse
from utils import *
import time

def parse_argument():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url', type=str,required=False, default='localhost:8001', help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source',type=str,required=True)
    parser.add_argument('--video_save_path',type=str,default="",required=False)
    opt=parser.parse_args()
    return opt


if __name__ == '__main__':
    opt=parse_argument()

    model_name=None
    labels={}
    if opt.model=="yolov5":
        model_name="yolov5_trt"
        model_version = "1"
        input_name="images"
        output_name="output0"
        c,w,h=3,640,640
        with open("labels/yolo_coco_labels.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                id,clx=line.split(":")
                labels[int(id)]=clx[1:-1]
    else:
        raise BaseException("Not supported model")
    
    # Create gRPC stub for communicating with the server
    MAX_BATCH_SIZE=8
    channel_opt = [('grpc.max_send_message_length', MAX_BATCH_SIZE*4 * c * w * h),('grpc.max_receive_message_length', MAX_BATCH_SIZE*4 * c * w * h)]
    channel = grpc.insecure_channel(opt.url,options=channel_opt)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

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
        video=cv2.VideoWriter(opt.result_save_path,fourcc,fps,(width,height))
    cur_id=0
    while cap.isOpened():
        success,frame=cap.read()
        if not success:
            break
        start_time=time.time()
        
        # Setting request
        request = service_pb2.ModelInferRequest()
        request.model_name = model_name
        request.model_version = model_version
        request.id = str(cur_id)
        cur_id+=1

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = input_name
        input.datatype = "FP32"
        input.shape.extend([1, c, w, h])
        request.inputs.extend([input])

        preprocessed_image,rs_image = input_preprocess(frame,opt.model)
        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = output_name
        request.outputs.extend([output])
        request.raw_input_contents.extend([preprocessed_image.tobytes()])

        response = grpc_stub.ModelInfer(request)
        frame=postprocess(rs_image,InferResult(response).as_numpy(output_name),labels,opt.model)
        frame = cv2.resize(frame, (width, height))
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


