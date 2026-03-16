conda create -n owdetr python=3.13 -y
conda activate owdetr
conda install cuda-toolkit=12.8 -y
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd models/ops && . make.sh
python setup.py clean --all
# python test.py

cd ../../
wget https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth -P ./models/cache/