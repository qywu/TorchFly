# FlyConfig

Flyconfig is a fast hierachical config system

## Example 1

You can use `--config` to specify the config path or set `config_path` in `FlyConfig.load(config_path=example1.yml)`.

```bash
python main.py  --config config/example1.yml 
```

The config should look like below:

```yaml
task:
  name: example1
  path: /home/user

```

## Example 2

A powerful design of `FlyConfig` is to load sub-configurations. For example, in `example2.yml`, we can define a list of subconfigs. It will load those configs in the corresponding directories and search for matching sub-configuration files.

```yaml
task:
    name: example2
subconfigs:
    - model: model_base
    - training: 1gpu
```

The final config looks like:

```yaml
task:
  name: example2
model:
  num_layers: 4
  hidden_size: 128
training:
  num_nodes: -1
  num_gpus_per_node: 2
  batch_size: 16
```

You can override those configs in the command-line argument.

```bash
python main.py  --config config/example2.yml model=model_large
```


The output looks like:

```yaml
task:
  name: example2
model:
  num_layers: 8
  hidden_size: 256
training:
  num_nodes: -1
  num_gpus_per_node: 2
  batch_size: 16
```


Or you can override a specific item in the config.

```bash
python main.py  --config config/example2.yml model=model_large model.num_layers=4 training.batch_size=128
```

```yaml
task:
  name: example2
model:
  num_layers: 4
  hidden_size: 256
training:
  num_nodes: -1
  num_gpus_per_node: 2
  batch_size: 128
```
