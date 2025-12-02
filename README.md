# EBRNet
EBRNet model: Lightweight Enhanced Bidirectional Recurrent Network for Satellite Video Super-Resolution


Install
1. Clone the repo
   git clone https://github.com/dz0602pshr/EBRNet.git
2. Install dependent packages
   cd EBRNet
   pip install -r requirements.txt
3. Install BasicSR
    python setup.py develop

Dataset Preparation
Download the SAT-MTB-VSR dataset from zenodo or Baidu Netdisk and unzip it to datasets/SAT-MTB-VSR/.
#We recommend using the following command to convert the files to lmdb format to speed up training:
python scripts/data_preparation/create_lmdb.py

Pretrained Models
Download the pretrained models from 链接: https://pan.baidu.com/s/1bImRcNeZWRmYvOgZaTB3hQ?pwd=wthe and put them in experiments/pretrained_models/SAT-MTB-VSR/.

Test
1. Single GPU
python basicsr/test.py -opt options/test_EBRNet.yml
2. Multiple GPU
CUDA_VISIBLE_DEVICES=0,1 ./scripts/dist_test.sh 2 options/test_EBRNet.yml
