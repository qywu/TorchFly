from omegaconf import OmegaConf

def save_config(config, filename):
    with open(filename, "w") as f:
        OmegaConf.save(config, f)    