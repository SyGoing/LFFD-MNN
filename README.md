[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

## the C++ implemententation of LFFD with MNN 
  I have implemented the LFFD  referring to the official python implementation
  
  paper:[LFFD: A Light and Fast Face Detector for Edge Devices](https://arxiv.org/abs/1904.10633)
  
  official github: [LFFD](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)
  
  My ncnn implementation is [here](https://github.com/SyGoing/LFFD-with-ncnn)
  
  My OpenVINO [implementation](https://github.com/SyGoing/LFFD-OpenVINO)
## some tips
 * You can set the input tensor shape smaller ,since you need to reduce the memory and accelerate the inference.
 * You can set the scale_num=8 to use another larger model. 
 * I just test it on vs2019 PC and the result is correct compared to original implementation,you can use the code to another device such as android、RK3399、and so on.

## how to convert the original model to mnn
  The original mxnet model has merged the preporcess(means and norms) and the detection output tensor has been sliced with the mxnet slice op in the symbol ,which caused convert failure.
so,you need to remove these ops ,in that way you can convert the model to onnx successfully.I will show you how to do that step by step, so when you train the model by yourself,
you can convert to your own model to onnx , and do more things.

  * First ,follow the author's original  github to build the devolopment environment.
  
  * Modify symbol_10_320_20L_5scales_v2.py (your_path/A-Light-and-Fast-Face-Detector-for-Edge-Devices\face_detection\symbol_farm) 
  
      in function loss_branch,Note out（注释掉） the line 57(predict_score = mxnet.symbol.slice_axis(predict_score, axis=1, begin=0, end=1)
	  
	  in function get_net_symbol, Note out（注释掉）the line 99(data = (data - 127.5) / 127.5,preprocess).
	  
  * Next,in this path , by doing "python symbol_10_320_20L_5scales_v2.py	",generate the symbol.json. symbol_10_560_25L_8scales_v1.py do the same thing .
  
  * To generate onnx model, cd your_path\A-Light-and-Fast-Face-Detector-for-Edge-Devices\face_detection\deploy_tensorrt
    python to_onnx.py 
	by doing this, you can find the generated onnx model in your_path\A-Light-and-Fast-Face-Detector-for-Edge-Devices\face_detection\deploy_tensorrt\onnx_files
	
  * In the last, you can use the MNN's MNNConvert to convert the model. have fun!
## test
 cd LFFD-MNN 
 mkdir build 
 cd build
 cmake ..
 make
## result show

![demo_res.jpg](https://raw.githubusercontent.com/SyGoing/LFFD-MNN/master/data/demo_res.jpg)

![selfie_res](https://raw.githubusercontent.com/SyGoing/LFFD-MNN/master/data/selfie_res.jpg)

![test_5_res](https://raw.githubusercontent.com/SyGoing/LFFD-MNN/master/data/test_5_res.jpg)

## TODO(you can refer this implementation to do more)
 - [x] MNN finished
 - [x] NCNN finished
 - [x] openvino demo: mxnet model-->onnx-->openvino 
 - [x] TensorRT demo: mxnet model --> onnx-->trt engine(finished and coming soon)
  
  
  
