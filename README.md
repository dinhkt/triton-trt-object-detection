# triton-trt-object-detection
Deployment of object detection models on Triton inference server with TensorRT backend

### Prepare the TRT engines
Prepare the TRT engines, create the config file and put in ```model_deloy/``` folder.

When generating the TRT engine, use the same TensorRT version as the TensorRT version of your Triton server. For example, 
I use ```nvcr.io/nvidia/tritonserver:22.08-py3``` for the Triton server, so I also use ```trtexec``` in ```nvcr.io/nvidia/tritonserver:22.08-py3-sdk``` to generate the engines. 
### Run Triton server: 

```cd triton-trt-object-detection/```

```docker run --gpus=all --rm --net=host -v ${PWD}/model_deploy:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --disable-auto-complete-config```

### Run client:
Requires installing ```opencv-python``` and ```tritonclient[all]``` packages by pip in advance .
You can choose to run http_client or grpc_client, in my experience grpc is about 20% faster than http client. 

Input source can be image file, video file or stream ( url or ```0``` for your webcam). To run the client:  

``` python grpc_client.py --source sample_inputs/vid.mp4 --mode yolov5 ```

Use ```--url``` to specify the URL of Triton inference server (default ```localhost:8000/8001``` for http/grpc service), ```--video_save_path``` to save the result video.

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmEyYWQ1MTNkMzRiODQ3NTM4MTI2OGNjOWFjNDU0MjFhZGQ0Njk2MCZjdD1n/cxnRUqlTZopXisDpff/giphy.gif)
