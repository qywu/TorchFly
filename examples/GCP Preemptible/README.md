<!-- 
```bash
gcloud compute instances create [INSTANCE_NAME] --preemptible
```

Handling preemption with a shutdown script might be a good solution.

https://cloud.google.com/compute/docs/instances/create-start-preemptible-instance

However, since our model size is often extremely large, the script might not finish before full shutdown. Therefore, we need other solutions.

https://cloud.google.com/compute/docs/instance-groups -->



# Setup for Google Cloud Instance

## Installation


### 1. Create Allow Network Traffic

```bash
# the most often ports http:80,443 ssh:22
gcloud compute firewall-rules create qywu-network-rules --allow tcp:80,tcp:443,tcp:22,tcp:8080
```

```bash
docker build -t qingyangwu/torchfly .

docker login qingyangwu

docker push qingyangwu/torchfly
```

```bash

docker run --runtime=nvidia --ipc=host -it --rm qingyangwu/torchfly

```



### To list available Nvidia GCP Images

```bash
gcloud compute images list --project=nvidia-ngc-public
```

### Debug Machine

```bash
gcloud compute instances create qywu-torchfly \
    --image nvidia-gpu-cloud-image-pytorch-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-b \
    --custom-vm-type n1 \
    --custom-cpu 4 \
    --custom-memory 4096MB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --tags qywu-network-rules \
    --preemptible 
```

### Real Machine

```bash
gcloud compute instances create qywu-torchfly \
    --image nvidia-gpu-cloud-image-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-b \
    --custom-vm-type n1 \
    --custom-cpu 4 \
    --custom-memory 4096MB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --tags qywu-network-rules \
    --preemptible 
```

## Delete GCP instance

```bash
gcloud compute instances delete torchfly
```


gcloud compute instances describe qywu-torchfly

gcloud compute instances start qywu-torchfly


https://docs.nvidia.com/ngc/ngc-gcp-vmi-release-notes/index.html