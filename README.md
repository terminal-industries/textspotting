# texspotting


1. TransformerTxtSpotting based on deepSolo, 

installation:
git clone https://github.com/ViTAE-Transformer/DeepSolo.git
cd DeepSolo
conda create -n deepsolo python=3.8 -y
conda activate deepsolo
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop


evaluation
python tools/train_net.py --config-file ./configs/R_50/IC15/finetune_150k_tt_mlt_13_15.yaml --eval-only MODEL.WEIGHTS ./output/R50/150k_tt_mlt_13_15/finetune/ic15/model_0002999.pth 