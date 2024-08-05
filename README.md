### README

#### 1. Start the Environment


**Hardware Environment**: g5.2xlarge EC2


First, log in to Docker, pull the image, and run the container:

```sh
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 418829965073.dkr.ecr.us-west-2.amazonaws.com
docker pull 418829965073.dkr.ecr.us-west-2.amazonaws.com/deepsolo-deployment-str-0.1.0:latest
docker run --gpus all --net=host -it -v /home/:/home/ 418829965073.dkr.ecr.us-west-2.amazonaws.com/deepsolo-deployment-str-0.1.0:latest /bin/bash
```

**Note**: Sometimes, TensorRT might not be found due to an environment setup issue. Follow these steps to reinstall TensorRT:

**Install TensorRT 8.6.1.6**:
Download the installation package from:
`s3://terminal-dev-bucket/model-artifacts/str/textspotting-v0.1.0/packages/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz`

**Uninstall the current TensorRT**:
```sh
pip uninstall tensorrt
```

**Install the new version**:
```sh
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1.6-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```

#### 2. Download the Models

The difference between these two models is that `0.1.0` is for detecting and recognizing key-ids only, while `0.1.1` is for recognizing all text in the images.

- `s3://terminal-dev-bucket/model-artifacts/str/textspotting-v0.1.0/exp_models/txtspotting_r50_v0.1.0.pth`
- `s3://terminal-dev-bucket/model-artifacts/str/textspotting-v0.1.0/exp_models/txtspotting_r50_v0.1.1.pth`

#### 3. Install DeepSolo-ONNX

Run the following command to complete the installation of DeepSolo-ONNX:

```sh
sh compile.sh
```

#### 4. Run ONNX + TRT Combined Conversion

Open `contents/convert2trt.py` and modify the following key fields:

- **OPSET_VER**:
   ```python
   OPSET_VER = 17
   ```
   - **Explanation**: This is the opset version for the ONNX model. Each version of the opset includes new operators and/or improvements to existing ones. Opset version 17 is relatively new and supports more features and operations.

- **IMAGE_PATH**:
   ```python
   IMAGE_PATH = "entry_mar_14_2024_01_00_53_pm_mar_14_2024_01_01_09_pm_fps_5_frame_34.png"
   ```
   - **Explanation**: This is the path to the input image for inference. This image will be used for testing during the model conversion and inference.

- **INTER_ONNX**:
   ```python
   INTER_ONNX = "/tmp/test_onnx_tmp.onnx"
   ```
   - **Explanation**: This is the path to the intermediate ONNX model file. This file is the initial ONNX model exported from the PyTorch model and is used for further simplification and conversion.

- **CONFIG**:
   ```python
   CONFIG = "config/Base_det_export_R50.yaml"
   ```
   - **Explanation**: This is the path to the configuration file. The configuration file contains various settings and hyperparameters for the model, such as network structure and pre-trained weight paths.

- **OUTPATH**:
   ```python
   OUTPATH = '/tmp/intermediate.onnx'
   ```
   - **Explanation**: This is the output path for the simplified ONNX model file. The ONNX model, after the simplification process, will be saved to this path.

- **TRT_ENGINE**:
   ```python
   TRT_ENGINE = '/home/ubuntu/models_deliver/txtspotting_r50_v0.1.1_2K.engine'
   ```
   - **Explanation**: This is the output path for the TensorRT engine file. After the TensorRT conversion, the model will be saved as this engine file, which can be used for efficient inference.

- **CHECKPOINT**:
   ```python
   CHECKPOINT = "/home/ubuntu/model_0319999_more_data.pth"
   ```
   - **Explanation**: This is the path to the PyTorch model checkpoint file. This file contains the trained model weights and is used for loading the model for export.

- **DIMS**:
   ```python
   DIMS = (2048, 1152)
   ```
   - **Explanation**: These are the dimensions (width and height) of the input image. The input image will be resized to these dimensions during preprocessing and model inference.

Then run the following command:

```sh
python3 contents/convert2trt.py
```

If successful, you will see the following output:

```
[W] --trt-config-func-name is deprecated and will be removed in Polygraphy 0.50.0. Use the config script argument instead.
[W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[W] onnx2trt_utils.cpp:400: One or more weights outside the range of INT32 was clamped
[I] Building engine with configuration:
    Flags                  | [TF32]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 16384.00 MiB, TACTIC_DRAM: 22512.75 MiB]
    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
    Profiling Verbosity    | ProfilingVerbosity.LAYER_NAMES_ONLY
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 133.999 seconds

DEBUG:root:stderr: 
```

This indicates that the ONNX and TRT conversions have both been successfully executed.

#### 5. Run Demo to Verify Success

Open the `demo.py` file and modify the `TRT_MODEL_PATH` to point to the successfully converted TRT model:

```python
TRT_MODEL_PATH = '/home/ubuntu/models_deliver/txtspotting_r50_v0.1.1_2K.engine'
```

Then run the following command:

```sh
cd ./contents/
python3 demo.py
```

You should see the output and detection results in the `output_image.png` file in the directory, and the average inference time per frame will also be printed.