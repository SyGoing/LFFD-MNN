## the C++ implemententation of LFFD with MNN 
  I have implemented the LFFD  referring to the official python implementation
  
  paper:[LFFD: A Light and Fast Face Detector for Edge Devices](https://arxiv.org/abs/1904.10633)
  
  official github: [LFFD](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)
  
  Also,my ncnn implementation is [here](https://github.com/SyGoing/LFFD-with-ncnn)
## some tips
 * You can set the input tensor shape smaller ,since you need to reduce the memory and accelerate the inference.
 * You can set the scale_num=8 to use another larger model. 
 * I just test it on vs2019 PC and the result is correct compared to original implementation,you can use the code to another device such as android、RK3399、and so on.
## TODO(you can refer this implementation to do more)
 * TensorRT demo: mmxnet model --> onnx-->trt engine
 * openvino demo: mxnet model-->onnx-->openvino
  
  