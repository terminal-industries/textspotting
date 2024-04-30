Demo

1. **Download the Latest Model Weights**  
   Acquire the latest weights from this location:
   ```
   s3://terminal-dev-bucket/data/tmpdata/model_str/R50/150k_tt_mlt_13_15/pretrain/model_0289999.pth
   ```

2. **Set Up the `transformertxtspotting` Environment**  
   Navigate to the `transformertxtspotting` directory and execute the following commands:
   ```
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -r requirements.txt
   python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
   python setup.py build develop
   ```
   **Note**: I have a pre-configured Docker container that can be run as follows:
   ```
   docker run --gpus all --net=host -it -v /home/:/home/ 418829965073.dkr.ecr.eu-west-2.amazonaws.com/ti_ml_pipeline:deepsolo_v1 /bin/bash 
   ```

3. **Run the Demo Script**  
   In the root directory, execute the following command to run the demo:
   ```
   python3 demo/demo.py --config-file ./demo/on_site_model.yaml --input /home/ubuntu/workspace/data/benchmark3.5/images/ --output ./TransformerTxtSpotting/output_res/ --opts MODEL.WEIGHTS model_0289999.pth
   ```

4. **Prediction Output**  
   The output from the prediction script will look something like this:
   ```
   [{'polygon': [(2306.08, 1274.87), (2304.97, 1232.07), (2363.13, 1230.56), (2364.24, 1273.36), (2306.08, 1274.87)], 'text': '1203'}]
   ```

5. **Benchmarking the Model**  
   To measure the performance of the trained model, follow these steps:
   5.1 **Download the Benchmark Data**:
   ```
   aws s3 sync s3://terminal-dev-bucket/training_data/ryder-on-site-16-03-test 
   ```
   5.2 **Execute the Benchmark**:
   ```
   python3 demo/benchmark.py --config-file ./demo/on_site_model.yaml --input /home/ubuntu/workspace/data/benchmark3.5/ --output ./TransformerTxtSpotting/output_res/ --opts MODEL.WEIGHTS /home/ubuntu/workspace/data/benchmark3.5/R_50/150k_tt_mlt_13_15/pretrain/model_0259999.pth
   ```

