# TorchFly

TorchFly is a PyTorch Fast Development Kit. The purpose is to learn the pipelines of the SOTA algorithms in Deep Learning areas like CV, NLP and RL. The utilities provided in this kit will shorten the time needed to rebuild some basic functions. Now, the kit is mainly for personal use, but will be updated from time to time. 

## Installation

You would need [apex](https://github.com/qywu/apex), but it requires some tiny modifications if you are using CUDA 10.1.
```
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



<!-- has bug
### Installing via pip: (Not supported for now)
Installing is simple using `pip`.

   ```bash
   pip install torchfly
   ```
-->
   
### Installing from source
You can clone the repository.
```
git clone https://github.com/qywu/TorchFly.git
cd torchfly
pip install -e .
```



## TODOS
 
1. Custom Bucket Sampler

2. Custom Beam Search

 
 ## Code References
 
 [FastAI](https://github.com/fastai)
 
 [AllenNLP](https://github.com/allenai/allennlp/)
 
 [Pytorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
