import abc


class AbstractLogger(abc.ABC):
    @abc.abstractmethod
    def notify(self, key, x, step):
        pass

    @abc.abstractmethod
    def close(self):
        pass
