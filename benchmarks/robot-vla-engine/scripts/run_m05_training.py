import os

os.environ.pop("LD_LIBRARY_PATH", None)
import argparse
import json

from robot_vla.m05_training import train_job

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
a = p.parse_args()
print(json.dumps(train_job(json.load(open(a.config))), indent=2))
