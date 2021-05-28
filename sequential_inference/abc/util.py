import abc


class Logger(abc.ABC):
    @abc.abstractmethod
    def notify(self, key, x, step):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class AbstractDataSampler(abc.ABC):
    @abc.abstractmethod
    def get_next(self, batch_size: int):
        pass
