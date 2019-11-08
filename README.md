# TorchFly

TorchFly is a PyTorch Fast Development Kit. The purpose is to learn the pipelines of the SOTA algorithms in Deep Learning areas like CV, NLP and RL. The utilities provided in this kit will shorten the time needed to rebuild some basic functions. Now, the kit is mainly for personal use, but will be updated from time to time. 

## Installation

[apex](https://github.com/qywu/apex) is required, but it may need modifications if cuda version is mismatched.
```bash
# modified the error due to cuda version
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



<!-- has bug
### Installing via pip: (Not supported for now)
Installing is simple using `pip`.

```
pip install torchfly
```

-->
   
### Installing from source
You can clone the repository.
```
git clone https://github.com/qywu/TorchFly.git
cd TorchFly
pip install -e .
pip install -r requirements.txt
```



## TODOS
 
1. Custom Bucket Sampler

2. Custom Beam Search


[] Remove [Allennlp]() and [transformers]() dependencies
[] Warum up Scheduler
 

## Code References
 
[FastAI](https://github.com/fastai)
 
[AllenNLP](https://github.com/allenai/allennlp/)
 
[Pytorch BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
