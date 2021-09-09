# ADG Steganography
This repository is our implementation of ADG.

**Provably Secure Generative Linguistic Steganography** (ACL 2021 *findings*)  
*Siyu Zhang, Zhongliang Yang, Jinshuai Yang, Yongfeng Huang*  
[https://arxiv.org/abs/2106.02011](https://arxiv.org/abs/2106.02011)


## Requirements
- PyTorch
- Numpy
- Gensim


## Datasets
1. Download training data from 
   [Google Drive](https://drive.google.com/drive/folders/1cLNaXsr1Wmim4NDQJMWqUQCZpigca3jO?usp=sharing) 
   or [百度网盘](https://pan.baidu.com/s/13EsaOcUIgD8YypjgY92Ikw) (提取码：udm4).
2. Place the datasets inside the `data/` folder.


## Training Language Models
You can either train from scratch or directly download checkpoints.

### Training from scratch
```
python run_lm.py
```
The checkpoint files will be saved into `models/` folder.

### Download Checkpoints
1. Download checkpoint files from 
   [Google Drive](https://drive.google.com/drive/folders/1PaMf7vT9sWl-EByUGZOuXfDilN4zcM7w?usp=sharing) 
   or [百度网盘](https://pan.baidu.com/s/1mIWB5XGTA-qKgTIm3zLtdQ) (提取码：7xcr).
2. Place the checkpoint files inside the `models/` folder.


## Generating Stegotext
```
python adg-stega.py
```
The generated stegotext will be saved into `stego/` folder.


## Quantitative Study
### Embedding Rate
```
cd measure/er
python measure_er.py
```
Results will be saved into `measure/er/er.json`.

### KL Divergence 1
```
cd measure/kld1
python measure_kld1.py
```

### KL Divergence 2
```
cd measure/kld2
python dataset_remove_unk.py
python build_vec.py
python measure_kld2.py
```
Results will be saved into `measure/kld2/kld2.json`.


## Citation
If you find our repository useful, please consider citing our paper:
```
```

## Acknowledgements
KLD2 is largely based on [sentence2vec](https://github.com/klb3713/sentence2vec).
