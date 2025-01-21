import argparse
import contextlib
import logging
import os
import sys
import subprocess
import signal
import requests
import json
import urllib.request
import urllib.parse
import io
from websocket import WebSocket
import uuid

INITIAL_PORT = int(os.getenv('INITIAL_PORT')) if os.getenv('INITIAL_PORT') is not None else 7860
API_COMMAND_LINE = os.getenv('API_COMMAND_LINE') if os.getenv('API_COMMAND_LINE') is not None else 'python /root/ComfyUI/main.py --highvram --fast '
MAX_COMFY_START_ATTEMPTS = int(os.getenv('MAX_COMFY_START_ATTEMPTS')) if os.getenv('MAX_COMFY_START_ATTEMPTS') is not None else 10
API_URL = os.getenv('API_URL') if os.getenv('API_URL') is not None else '127.0.0.1'

class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}"
        return True


class FeatureServerInterface:
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModuleInterface, cls).__new__(cls)
        return cls._instance

    def find_available_port(self): # If the initial port is already in use, this method finds an available port to start the API server on
        port = INITIAL_PORT
        while True:
            try:
                response = requests.get(f'http://{API_URL}:{port}')
                if response.status_code != 200:
                    return port
                else:
                    port += 1
            except requests.ConnectionError:
                return port

    def start_api(self): # This method is used to start the API server
        if not self.is_api_running(): # Block execution until the API server is running
            api_command_line = API_COMMAND_LINE + f" --port {self.urlport}" # Add the port to the command line
            if self._process is None or self._process.poll() is not None: # Check if the process is not running or has terminated for some reason
                self._process = subprocess.Popen(api_command_line.split())
                print("API process started with PID:", self._process.pid)
                attempts = 0
                while not self.is_api_running(): # Block execution until the API server is running
                    if attempts >= MAX_COMFY_START_ATTEMPTS:
                        raise RuntimeError(f"API startup procedure failed after {attempts} attempts.")
                    time.sleep(COMFY_START_ATTEMPTS_SLEEP)  # Wait before checking again, for 1 second by default
                    attempts += 1 # Increment the number of attempts
                print(f"API startup procedure finalized after {attempts} attempts with PID {self._process.pid} in port {self.urlport}")
                time.sleep(1.5)  # Wait for 0.5 seconds before returning

    def is_api_running(self): # This method is used to check if the API server is running
        test_payload = TEST_PAYLOAD
        try:
            print(f"Checking web server is running in {self.server_address}...")
            response = requests.get(self.server_address)
            if response.status_code == 200: # Check if the API server tells us it's running by returning a 200 status code
                self.ws.connect(self.ws_address)
                print(f"Web server is running (status code 200). Now trying test image...")

                # TODO  # Check if the API server is actually running by sending a test request.

                if test_image is not None:  # this ensures that the API server is actually running and not just the web server
                    return True
                return False
        except Exception as e:
            print("API not running:", e)
            return False










def main(args, extras) -> None:
    # set CUDA_VISIBLE_DEVICES if needed, then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]
    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    if args.typecheck:
        from jaxtyping import install_import_hook
        install_import_hook("sgps", "typeguard.typechecked")
        
    import sgps
    from sgps.systems.base import BaseSystem
    from sgps.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from sgps.utils.config import ExperimentConfig, load_config
    from sgps.utils.misc import get_rank, time_recorder
    from sgps.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.benchmark:
        time_recorder.enable(True)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))        

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    dm = sgps.find(cfg.data_cls)(cfg.data)
    system: BaseSystem = sgps.find(cfg.system_cls)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
    rank_zero_only(
        lambda: os.makedirs(os.path.join(cfg.trial_dir, "save"), exist_ok=True)
    )()

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(
            #     os.path.join(cfg.trial_dir, "code"), use_version=False
            # ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [
                ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
            ]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        ]
        if args.wandb:
            wandb_logger = WandbLogger(project="TPA", name=f"{cfg.name}-{cfg.tag}")
            system._wandb_logger = wandb_logger
            loggers += [wandb_logger]
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()

    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])
    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.validate(system, datamodule=dm)
        # trainer.test(system, datamodule=dm)
        if args.gradio:
            # also export assets if in gradio mode
            trainer.predict(system, datamodule=dm)
    elif args.validate:
        # manually set epoch and global_step as they cannot be automatically resumed
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="if true, log to wandb")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="if true, set to benchmark mode to record running times",
    )

    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()

    if args.gradio:
        # FIXME: no effect, stdout is not captured
        with contextlib.redirect_stdout(sys.stderr):
            main(args, extras)
    else:
        main(args, extras)