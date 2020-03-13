import os
import sys
import hydra
import time
import datetime
import subprocess
import regex as re
import logging

logger = logging.getLogger(__name__)

def check_instance_preemptible(instance_name):
    output = subprocess.run(
        f"gcloud compute instances describe {instance_name}", shell=True, check=True, stdout=subprocess.PIPE
    )
    status = re.findall(r"(?<=preemptible: ).*", output.stdout.decode("utf-8"))[0]
    logger.info(f"{instance_name} is {status}.")
    return 0


def check_instance_status(instance_name):
    output = subprocess.run(
        f"gcloud compute instances describe {instance_name}", shell=True, check=True, stdout=subprocess.PIPE
    )
    status = re.findall(r"(?<=status: ).*", output.stdout.decode("utf-8"))[0]
    logger.info(f"{instance_name} is {status}.")
    return status


def start_instance(instance_name):
    output = subprocess.run(
        f"gcloud compute instances start {instance_name}", shell=True, check=True, stdout=subprocess.PIPE
    )
    logger.info(output.stdout.decode("utf-8"))
    return 0


def stop_instance(instance_name):
    output = subprocess.run(
        f"gcloud compute instances stop {instance_name}", shell=True, check=True, stdout=subprocess.PIPE
    )
    logger.info(output.stdout.decode("utf-8"))
    return 0


def main_loop(cfg):
    # report preemptible
    check_instance_preemptible(cfg.preempt.instance_name)

    last_time = datetime.datetime.strptime(cfg.preempt.start_time, '%Y-%m-%d %H:%M')
    elapsed_time = last_time - last_time
    logger.info(f"Start time: {last_time}")

    while elapsed_time.total_seconds() < cfg.preempt.task_duration * 3600:

        status = check_instance_status(cfg.preempt.instance_name)
        logger.info(f"Checking Status: {status}")

        # wait 30s if it is stopping
        if status == "STOPPING":
            time.sleep(30)
            start_instance(cfg.preempt.instance_name)
            logger.info(f"Starting Instance {cfg.preempt.instance_name}")
        elif status == "TERMINATED":
            start_instance(cfg.preempt.instance_name)
            logger.info(f"Starting Instance {cfg.preempt.instance_name}")
        elif status == "RUNNING":
            pass

        time.sleep(cfg.preempt.loop_interval)
        elapsed_time = datetime.datetime.now() - last_time
        # print(elapsed_time.total_seconds())

    stop_instance(cfg.preempt.instance_name)
    logger.info("The job finished!")


@hydra.main(config_path='conf/config.yaml')
def main(cfg=None):
    print(cfg.pretty())
    main_loop(cfg)


if __name__ == "__main__":
    main()