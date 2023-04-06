# triton-trt-object-detection
Deployment of quantized TRT object detection models on Triton inference server

### Prepare the TRT engines 
The engine building code is based on https://github.com/NVIDIA/TensorRT/tree/main/samples/python/efficientdet, with modifications to fit with TensorRT8 and change the preprocess function in INT8 calibrator for Yolo and Resnet models.

Prepare the ONNX models in advance. For example, ```yolov5s.onnx``` model can be exported by using ```export.py``` in ```ultralytics/yolov5```.
To build the engine and post-training-quantization (PTQ) model, go to ```build_quantized_engine/``` and execute:
#### For FP16:
  ```python build_engine.py --onnx yolov5s.onnx --engine yolov5s_fp16.trt --precision fp16```
#### For INT8:
  ```python build_engine.py --onnx yolov5s.onnx --engine yolov5s_int8.trt --precision int8 --calib_input path/to/sample/datasets --calib_preprocessor yolo```

with ```path/to/sample/datasets``` is the path to the directory storing the sample images for INT8 calibration.

Change to engine name to ```model.plan```, create the config file and put in ```model_deloy/``` folder with following this structure:
```
model_deploy/
    yolov5_trt/
        1/
          model.plan
        config.pbtxt
```

When generating the TRT engine, use the same TensorRT version as the TensorRT version of your Triton server that you will deploy the model. For example, my tensorrt version is 8.5.1.7, so I use ```nvcr.io/nvidia/tritonserver:22.12-py3``` for Triton server, according to https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-22-12.html#rel-22-12

#### (Example) Quantization results of Torchvision Resnet50 (on NVIDIA RTX 3060):
```
Run python ./test_inference/test_resnet50_inference.py 

Model         |   Inference Time   | Accuracy (Imagenet/val2017)
-----------------------------------------------------------------
pytorch(fp32) |   5.87ms           |   74.46%
-----------------------------------------------------------------
TRT (fp16)    |   0.74ms           |   72.14%
-----------------------------------------------------------------
TRT (int8)    |   0.43ms           |   72.34%
-----------------------------------------------------------------
```
#### Quantization results of ultralytics/yolov5s (on NVIDIA RTX 3060):
```
Run python ./test_inference/test_yolov5_inference.py to generate the detection results and 
run python mAP/main.py -na -np to compute the mAP 50

Model         |   Inference Time   | mAP50 (COCO/val2017)
-----------------------------------------------------------------
pytorch(fp32) |   8.7ms            |   47.3
-----------------------------------------------------------------
TRT (int8)    |   1.12ms           |   28.65
-----------------------------------------------------------------
```

### Run Triton server: 

```cd triton-trt-object-detection/```

```docker run --gpus=all --rm --net=host -v ${PWD}/model_deploy:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --disable-auto-complete-config```

### Run client:
Requires installing ```opencv-python``` and ```tritonclient[all]``` packages by pip in advance .
You can choose to run http_client or grpc_client, in my experience, the end-to-end latency of grpc request is about 20% faster than http. 

Input source can be image file, video file or stream ( url or ```0``` for your webcam). To run the client:  

``` python grpc_client.py --source sample_inputs/vid.mp4 --model yolov5 ```

Use ```--url``` to specify the URL of Triton inference server (default ```localhost:8000/8001``` for http/grpc service), ```--video_save_path``` to save the result video.

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmEyYWQ1MTNkMzRiODQ3NTM4MTI2OGNjOWFjNDU0MjFhZGQ0Njk2MCZjdD1n/cxnRUqlTZopXisDpff/giphy.gif)

### Todo:
- [x] Yolov5
- [ ] Detection Transformer
