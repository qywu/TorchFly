# MNIST example


## Run

You can use `--config` to specify the config path or set `config_path` in `FlyConfig.load(config_path=example1.yml)`.

```bash
python main.py --config config/config.yaml
```

For distributed training, please use `torch.distributed.launch`. See the example below:

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --config config/config.yaml training.num_gpus_per_node=4
```


## Configurations

You can change configurations in the training to enable distributed training or mixed precision training.

```yaml
training:
    random_seed: 123
    fp16: True
    num_gpus_per_node: 2
    batch_size: 32
    gradient_accumulation_batches: 1
```

