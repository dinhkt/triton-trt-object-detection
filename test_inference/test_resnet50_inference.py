import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch


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
        return [out.host.reshape(batch_size,-1) for out in self.outputs],infer_time

def preprocess(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0 
    image[:,:,0] = (image[:,:,0]-0.485)/0.229
    image[:,:,1] = (image[:,:,1]-0.456)/0.224
    image[:,:,2] = (image[:,:,2]-0.406)/0.225
    image = np.transpose(image, [2, 0, 1])
    return np.expand_dims(image,0)

def test_int8trt_inference(engine_path):
    print("Run Test Tensorrt ....")
    model = TrtModel(engine_path)
    # warm-up
    dummy_inp=np.random.randint(0,255,(1,*model.engine.get_binding_shape(0)[1:]))/255.0
    for _ in range(20):
        model(dummy_inp)
    path = '../../datasets/imagenet1k/val/'
    f_labels = open("../../datasets/imagenet1k/labels1.txt",'r')
    labels = [x.split()[0] for x in f_labels.readlines()]
    directory = os.listdir(path)
    num = 0 
    correct = 0
    totalTime = 0
    for label in directory:
        files = os.listdir(path + label)
        for f in files:
            data = preprocess(path + label + '/' + f)
            outputs,t= model(data)
            pred=np.argmax(outputs[0])
            totalTime += t
            correct += int(pred==labels.index(label))
            num += 1
    inferenceTime = totalTime / num
    accuracy = (correct / num) * 100
    print("TRT INT8 Inference Time: ", inferenceTime,"ms")
    print("TRT INT8 Accuracy: ", accuracy, '%')

import torchvision
import torch
from PIL import Image
def test_torch_r50_inference():
    dev=torch.device("cuda")
    model=torchvision.models.resnet50(pretrained=True)
    model.eval()
    model.to(dev)
    torch_preprocess = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224,224)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # warm-up
    dummy_inp=torch.rand((1,3,224,224))
    for _ in range(20):
        model(dummy_inp.to(dev))
    path = '../../datasets/imagenet1k/val/'
    f_labels = open("../../datasets/imagenet1k/labels1.txt",'r')
    labels = [x.split()[0] for x in f_labels.readlines()]
    directory = os.listdir(path)
    num = 0 
    correct = 0
    totalTime = 0
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for label in directory:
        files = os.listdir(path + label)
        for f in files:
            img = Image.open(path + label + '/' + f).convert('RGB')
            data = torch_preprocess(img).unsqueeze(0)
            start.record()
            outputs= model(data.to(dev))
            end.record()
            torch.cuda.synchronize()
            pred=np.argmax(outputs[0].detach().cpu().numpy())
            totalTime += start.elapsed_time(end)
            correct += int(pred==labels.index(label))
            num += 1
    inferenceTime = totalTime / num
    accuracy = (correct / num) * 100
    print("PyTorch Inference Time: ", inferenceTime,"ms")
    print("PyTorch Accuracy: ", accuracy, '%')    


if __name__ == "__main__":
    test_int8trt_inference('resnet50_int8.trt')
    test_int8trt_inference('resnet50_fp16.trt')
    # test_torch_r50_inference()