import yaml
from easydict import EasyDict as edict
from utils.console_logger import ConsoleLogger
from tensorboardX import SummaryWriter

def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))
    return config

def add_log(configs):
    LOG = ConsoleLogger(configs.task, 'train')
    logdir = LOG.getLogFolder()
    LOG.info(configs)
    train_summary_writer = SummaryWriter(logdir, 'train')
    return logdir, LOG, train_summary_writer

class LearningRateLambda():
    def __init__(self, decay_schedule, *,
                 decay_factor=0.1,
                 decay_epochs=1.0,
                 warm_up_start_epoch=0,
                 warm_up_epochs=2.0,
                 warm_up_factor=0.01,
                 warm_restart_schedule=None,
                 warm_restart_duration=0.5):
        self.decay_schedule = decay_schedule
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warm_up_start_epoch = warm_up_start_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor
        self.warm_restart_schedule = warm_restart_schedule
        self.warm_restart_duration = warm_restart_duration

    def __call__(self, step_i):
        lambda_ = 1.0

        if step_i <= self.warm_up_start_epoch:
            lambda_ *= self.warm_up_factor
        elif self.warm_up_start_epoch < step_i < self.warm_up_start_epoch + self.warm_up_epochs:
            lambda_ *= self.warm_up_factor**(
                1.0 - (step_i - self.warm_up_start_epoch) / self.warm_up_epochs
            )

        for d in self.decay_schedule:
            if step_i >= d + self.decay_epochs:
                lambda_ *= self.decay_factor
            elif step_i > d:
                lambda_ *= self.decay_factor**(
                    (step_i - d) / self.decay_epochs
                )

        for r in self.warm_restart_schedule:
            if r <= step_i < r + self.warm_restart_duration:
                lambda_ = lambda_**(
                    (step_i - r) / self.warm_restart_duration
                )

        return lambda_