<img src="docs/images/torchfly.svg" width="300" >

--------------------------------------------------------------------------------

## Installment


Please download [conda](https://www.anaconda.com/distribution/#download-section) and create a virtual environment first.

```bash
# create virtual env
conda create -n torchfly python=3.6
```

[apex](https://github.com/qywu/apex) is required, but it may need modifications if cuda version is mismatched.

```bash
# make sure ``nvcc`` is installed
# modified the error due to cuda version
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Then install this library in developer mode:

```bash
# clone repo
git clone https://github.com/qywu/TorchFly

# install the repo
pip install -e .
```

To use the repo,

```python
import torchfly
```

## Documentation

https://qywu.github.io/TorchFly
