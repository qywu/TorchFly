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

Please go to [Google Cloud SDK](https://cloud.google.com/sdk/docs/downloads-apt-get?hl=tr) for the details of installation.

## Setup

### 1. Create Allow Network Traffic

```bash
# the most often ports http:80,443 ssh:22
gcloud compute firewall-rules create qywu-network-rules --allow tcp:80,tcp:443,tcp:22,tcp:8080,tcp:8888-8900,tcp:6006-7000
```

gcloud compute firewall-rules delete qywu-network-rules 

### 2. Create a Machine

```bash
gcloud compute instances create qywu-preemptible \
    --image nvidia-gpu-cloud-image-pytorch-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-b \
    --custom-vm-type n1 \
    --custom-cpu 32 \
    --custom-memory 208GB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --tags qywu-network-rules \
    --accelerator type=nvidia-tesla-v100,count=8 \
    --scopes cloud-platform \
    --preemptible 
```

If you only want to debug, change the accelerator to T4 first:

```bash
gcloud compute instances create qywu-preemptible \
    --image nvidia-gpu-cloud-image-pytorch-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-a \
    --custom-vm-type n1 \
    --custom-cpu 32 \
    --custom-memory 208GB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --tags qywu-network-rules \
    --accelerator type=nvidia-tesla-t4,count=4 \
    --scopes cloud-platform \
    --preemptible 
```

or non-gpu version

```bash
gcloud compute instances create qywu-preemptible \
    --image nvidia-gpu-cloud-image-pytorch-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-b \
    --custom-vm-type n1 \
    --custom-cpu 2 \
    --custom-memory 12GB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --scopes cloud-platform \
    --tags qywu-network-rules
```


### 3. Remtoe Setup

1. SSH to the remote server first, 
    ```
    gcloud beta compute ssh --zone "us-west1-a" "qywu-preemptible" --project "nlp-compute-project"
    ```

2. Then install gcsfuse with 
(https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md)

    ```bash
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

    sudo apt-get update
    sudo apt-get install gcsfuse
    ```

### 4. Define Autostart Script

On you local machine:

```bash
gcloud compute instances add-metadata qywu-pretrain\
    --metadata-from-file startup-script=startup_script.sh
```

Here is an example of `startup_script.sh`.

```bash
#! /bin/bash

# setup gs bucket
mkdir -p /data
mkdir -p /workspace/pretrain_roberta
gcsfuse roberta_processed_corpus /data
gcsfuse qywu-pretrain-roberta-bucket /workspace/pretrain_roberta

# run docker
docker run -d --ipc=host pretrain
```


## Others

### Docker Setup
```bash
docker build -t qingyangwu/torchflgcloud compute instances create qywu-pretrain-test2 \
    --image nvidia-gpu-cloud-image-pytorch-20191120 \
    --image-project nvidia-ngc-public \
    --zone us-west1-b \
    --custom-vm-type n1 \
    --custom-cpu 2 \
    --custom-memory 12GB \
    --boot-disk-size 32GB \
    --boot-disk-type pd-ssd \
    --scopes cloud-platform \
    --tags qywu-network-rulesy .

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


### Delete GCP instance

```bash
gcloud compute instances delete torchfly
```


gcloud compute instances describe qywu-torchfly

gcloud compute instances start qywu-torchfly


https://docs.nvidia.com/ngc/ngc-gcp-vmi-release-notes/index.html