#!/usr/bin/python
from subprocess import Popen, DEVNULL
from shlex import split
from pathlib import Path

import time
import os


class start_ray_cluster:
    def __init__(self, python_module: str = 'tensorflow/2.9.0'):
        self.python_module = python_module
        self.prom_image = 'prom/prometheus:v2.42.0'
        self.gf_image = 'grafana/grafana-oss:9.4.3'
        self._cluster_dir = f'{os.getenv("SCRATCH")}/ray_cluster/'
        self.prom_dir = f'{self._cluster_dir}prometheus'
        self.gf_dir = f'{self._cluster_dir}grafana'
        self.gf_root_url = f"https://jupyter.nersc.gov{os.getenv('JUPYTERHUB_SERVICE_PREFIX')}proxy/3000"
        self._start_up()
        
    def _start_up(self):
        self._setup()
        self.ray = self._start_ray()
        time.sleep(5)
        self.prom = self._start_prom()
        self.gf = self._start_gf()
        print("<> Cluster completed startup")
        
    def _setup(self):
        Path(f"{self.prom_dir}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.gf_dir}").mkdir(parents=True, exist_ok=True)
        
    def _start_ray(self):
        print("<> Starting Ray head node")
        return Popen(
            split(f'bash -c "module load {self.python_module};' \
                  f'export RAY_GRAFANA_IFRAME_HOST={self.gf_root_url};' \
                  f'ray start --head --block"')
        )
        
    def _start_prom(self):
        print("<> Starting Prometheus Service")
        return Popen(
            split(f'shifter --image={self.prom_image} --volume={self.prom_dir}:/prometheus ' \
                  f'/bin/prometheus ' \
                  f'--config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml ' \
                  f'--storage.tsdb.path=/prometheus'),
            stdout=DEVNULL, stderr=DEVNULL
        )

    def _start_gf(self):
        print("<> Starting Grafana Service")
        return Popen(
            split(f'shifter --image={self.gf_image} --volume={self.gf_dir}:/grafana ' \
                  f'--env GF_PATHS_DATA=/grafana ' \
                  f'--env GF_PATHS_PLUGINS=/grafana/plugins ' \
                  f'--env GF_SERVER_ROOT_URL={self.gf_root_url}/ ' \
                  f'--env GF_PATHS_CONFIG=/tmp/ray/session_latest/metrics/grafana/grafana.ini ' \
                  f'--env GF_PATHS_PROVISIONING=/tmp/ray/session_latest/metrics/grafana/provisioning ' \
                  f'--entrypoint &'),
            stdout=DEVNULL, stderr=DEVNULL
        )

    def kill(self):
        self.prom.kill()
        self.gf.kill()
        self.ray.kill()
        return