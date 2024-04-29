Please follow the steps below to install the necessary execution packages in our current Docker environment:

1. Download the latest weights from the following location:

```
s3://terminal-dev-bucket/data/tmpdata/model_str/R50/150k_tt_mlt_13_15/pretrain/model_0289999.pth
```

2. Navigate to the `TransformerTxtSpotting` directory:

```
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```

3. Execute the following command in the root directory:

```
python3 demo/demo.py --config-file ./demo/on_site_model.yaml --input /home/ubuntu/workspace/data/benchmark3.5/images/ --output ./TransformerTxtSpotting/output_res/ --opts MODEL.WEIGHTS model_0289999.pth
```

4. The output of the prediction is as follows:

```
[{'polygon': [(2306.08, 1274.87), (2304.97, 1232.07), (2363.13, 1230.56), (2364.24, 1273.36), (2306.08, 1274.87)], 'text': '1203'}]
```
