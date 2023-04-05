import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch
import sys
sys.path.append("../")
from triton_client.utils import *

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
    
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        f= open(engine_path, 'rb')
        engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
                        
    def allocate_buffers(self):        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
       
    def __call__(self,x:np.ndarray,batch_size=1): 
        start,end = cuda.Event(),cuda.Event()
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        start.record()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        end.record()
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        # end.record() 
        self.stream.synchronize()    
        infer_time=start.time_till(end)
        # return np.argmax(self.outputs)    
        return [out.host.reshape(batch_size,25200,85) for out in self.outputs],infer_time

def postprocess(outputs,orig_shape):
    res=[]
    orig_h,orig_w,_=orig_shape
    # print(orig_shape)
    if orig_h>orig_w:
        scale=orig_h/640
        pad_x=(640*scale-orig_w)/2
        pad_y=0
    else:
        scale=orig_w/640
        pad_y=(640*scale-orig_h)/2
        pad_x=0
    # print(scale,pad_x,pad_y)
    for ig in outputs:
        ig=NMS(ig,confThresh=0.2, overlapThresh=0.5)
        for pred in ig:
            lab=np.argmax(pred[5:])
            cfd_score=pred[4]
            x1,y1,x2,y2=pred[0]-pred[2]/2,pred[1]-pred[3]/2,pred[0]+pred[2]/2,pred[1]+pred[3]/2
            # print(x1,y1,x2,y2)
            x1=int(x1*scale-pad_x)
            x2=int(x2*scale-pad_x)
            y1=int(y1*scale-pad_y)
            y2=int(y2*scale-pad_y)
            res.append([lab,cfd_score,x1,y1,x2,y2])
    return res

def test_int8trt_inference(engine_path):
    print("Run Test Tensorrt ....")
    model = TrtModel(engine_path)
    # warm-up
    dummy_inp=np.random.randint(0,255,(1,*model.engine.get_binding_shape(0)[1:]))/255.0
    for _ in range(20):
        model(dummy_inp)
    path = '../../datasets/coco/val2017'
    labels={}
    with open("../triton_client/labels/yolo_coco_labels.txt","r") as f:
        lines=f.readlines()
        for line in lines:
            id,clx=line.split(":")
            labels[int(id)]=clx[1:-1]
    SAVE_DIR="../mAP/input/detection-results/"
    directory = os.listdir(path)
    num = 0 
    totalTime = 0
    for f in directory:
        img = cv2.imread(path + '/' + f)
        # img=cv2.imread("bear.jpg")
        fw=open(SAVE_DIR+f.split(".")[0]+".txt","w")
        data,_=input_preprocess(img)
        outputs,t= model(data)
        res=postprocess(outputs,img.shape)
        for oj in res:
            for each in oj:
                fw.write(str(each)+" ")
            fw.write("\n")
        fw.close()
        totalTime += t
        num += 1
        # break
    inferenceTime = totalTime / num
    print("TRT INT8 Inference Time: ", inferenceTime,"ms")
 

if __name__ == "__main__":
    test_int8trt_inference("yolov5_int8.trt")


