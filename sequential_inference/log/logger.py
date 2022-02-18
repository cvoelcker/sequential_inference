import glob
import shutil
import os
import pickle
from collections import defaultdict
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sequential_inference.abc.common import Checkpointable

from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import AbstractLogger


def mean_dict(lod):
    mean_d = {}
    for k in lod[0].keys():
        if torch.is_tensor(lod[0][k]):
            mean_d[k] = torch.stack([i[k] for i in lod], 0).mean().detach()
        else:
            mean_d[k] = torch.tensor([i[k] for i in lod]).mean().detach()
    return mean_d


class CmdLogger(AbstractLogger):
    def __init__(self, keys):
        self.keys = keys

    def notify(self, key, x, step):
        if key in self.keys:
            print(x)


class CmdAverageLogger(AbstractLogger):
    def __init__(self):
        self.d = []

    def notify(self, key, x, step):
        if key == "step":
            self.d.append(x)
        if key == "log":
            print(mean_dict(self.d))
            self.d = []


class FileLogger(AbstractLogger):
    def __init__(self, file_name="log.pkl", keys=("step",)):
        self.file_name = file_name
        self.keys = keys
        self.d = []

    def notify(self, key, x, step):
        if key in self.keys:
            self.d.append(x)

    def close(self):
        with open(self.file_name, "wb") as f:
            pickle.dump(self.d, f)


class TorchTensorboardHandler(AbstractLogger):
    def __init__(
        self,
        log_dir="tb_logs",
        name_dir="default",
        keys=("step", "epoch"),
        log_name_list=None,
        reset_logdir=True,
    ):
        self.keys = keys
        full_path = os.path.join(os.path.abspath(log_dir), name_dir)  # type: ignore
        self.logger = SummaryWriter(full_path, flush_secs=10)
        print(f"Logging to {full_path}")
        if reset_logdir:
            for the_file in os.listdir(full_path):
                file_path = os.path.join(full_path, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        self.step = 0
        self.log_name_list = log_name_list
        super().__init__()

    def register_logging(self, log_key):
        if self.log_name_list is None:
            self.log_name_list = [log_key]
        else:
            self.log_name_list.append(log_key)

    def notify(self, key, data, step):
        skip_key = not key in self.keys

        for tag, value in data.items():
            skip_tag = (self.log_name_list is not None) and (
                tag not in self.log_name_list
            )
            if skip_key or skip_tag:
                continue
            if type(value) is not torch.Tensor:
                try:
                    self.logger.add_scalar(tag, value.mean(), self.step)
                except Exception as e:
                    continue
            else:
                self.logger.add_scalar(
                    tag, value.detach().mean().cpu().item(), self.step
                )
        if "step" in data:
            self.step += data["step"]
        else:
            self.step += 1
        self.logger.flush()

    def reset(self):
        self.step = 0

    def close(self):
        self.logger.flush()
        self.logger.close()


class CsvLogger(AbstractLogger):
    def __init__(
        self, log_dir="raw_logs", name="default", keys=("log",), overwrite=False
    ):
        self.scalars_names = []
        self.scalars_names_is_full = False

        self.save_dir = log_dir

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.file_path = os.path.join(self.save_dir, f"{name}.csv")
        if overwrite:
            self.file = open(self.file_path, "w", buffering=1)
        else:
            self.file = open(self.file_path, "a", buffering=1)
        self.buffer = defaultdict(dict)

        self.keys = keys

    def notify(self, key, data, step):
        if key in self.keys:
            for name, value in data.items():
                if name not in self.scalars_names:
                    if not self.scalars_names_is_full:
                        self.scalars_names.append(name)
                    else:
                        raise Exception("try to log unknown scalar {}".format(name))
                self.buffer[step].update({name: value})
            self._flush()

    def _flush(self):
        if not self.scalars_names_is_full and os.stat(self.file_path).st_size == 0:
            line = "\t".join(["step"] + self.scalars_names)
            line += "\n"
            self.file.write(line)
            self.scalars_names_is_full = True

        steps = sorted(list(self.buffer.keys()))
        for step in steps:
            if len(self.buffer[step].keys()) == len(self.scalars_names):
                values = self.buffer.pop(step)
                line = "\t".join(
                    map(str, [step] + [values[k].item() for k in self.scalars_names])
                )
                line += "\n"
                self.file.write(line)

    def close(self):
        self.file.close()


class Checkpointing:
    def __init__(self, chp_dir=".", chp_name="checkpoints", overwrite=False):
        self.overwrite = overwrite

        self.chp_dir = os.path.join(chp_dir, chp_name)

        if overwrite:
            shutil.rmtree(self.chp_dir, ignore_errors=True)
            os.makedirs(self.chp_dir)
            self.counter = 0
        else:
            if not os.path.exists(self.chp_dir):
                os.makedirs(self.chp_dir)
            self.counter = self.get_num_saved()

    def __call__(self, checkpointable: Checkpointable):
        to_save = checkpointable.state_dict()
        if self.overwrite:
            path = os.path.join(self.chp_dir, f"save.torch")
        else:
            path = os.path.join(self.chp_dir, f"{self.counter:06d}.torch")
        self.counter += 1
        torch.save(to_save, path)
        return {}

    def get_latest(self) -> str:
        all_checkpoints = sorted(os.listdir(self.chp_dir))
        if len(all_checkpoints) == 0:
            raise ValueError("No checkpoint found")
        else:
            return os.path.join(self.chp_dir, all_checkpoints[-1])

    def get_num_saved(self) -> int:
        all_checkpoints = sorted(os.listdir(self.chp_dir))
        return len(all_checkpoints)
