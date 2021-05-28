import shutil
import os
import pickle
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import Logger


def mean_dict(lod):
    mean_d = {}
    for k in lod[0].keys():
        if torch.is_tensor(lod[0][k]):
            mean_d[k] = torch.stack([i[k] for i in lod], 0).mean().detach()
        else:
            mean_d[k] = torch.tensor([i[k] for i in lod]).mean().detach()
    return mean_d


class CmdLogger(Logger):
    def __init__(self, keys):
        self.keys = keys

    def notify(self, key, x, step):
        if key in self.keys:
            print(x)


class CmdAverageLogger(Logger):
    def __init__(self):
        self.d = []

    def notify(self, key, x, step):
        if key == "step":
            self.d.append(x)
        if key == "log":
            print(mean_dict(self.d))
            self.d = []


class FileLogger(Logger):
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


class TorchTensorboardHandler(Logger):
    def __init__(
        self,
        logdir="tb_logs",
        namedir="default",
        keys=("step", "epoch"),
        log_name_list=None,
        reset_logdir=True,
    ):
        self.keys = keys
        full_path = os.path.join(os.path.abspath(logdir), namedir)
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
        self.logger.close()

    def reset(self):
        self.step = 0


class CsvLogger(Logger):
    def __init__(self, save_dir, log_name, keys=("log",), overwrite=False):
        self.scalars_names = []
        self.scalars_names_is_full = False
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = os.path.join(save_dir, "raw_logs")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.file_path = os.path.join(self.save_dir, f"{log_name}.csv")
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
    def __init__(self, chp_dir, chp_name, counter=0, overwrite=False):
        self.counter = counter
        self.overwrite = overwrite

        self.chp_dir = chp_dir
        self.chp_name = chp_name

    def __call__(self, experiment: AbstractExperiment):
        to_save = experiment.state_dict()
        path = os.path.join(self.chp_dir, self.chp_name + f"_{self.counter:06d}.torch")
        self.counter += 1
        torch.save(to_save, path)
        return {}
