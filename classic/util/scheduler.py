import abc
from typing import List


class BaseScheduler(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    def get_learning_rate(self) -> float:
        return self.learning_rate
    
    @abc.abstractmethod
    def step(self, epoch: int, n_epochs: int) -> None:
        pass

class LearningRateDecayScheduler(BaseScheduler):
    def __init__(self, learning_rate: float = 1e-3, decay: float = 0.1, milestones: List[int] = []) -> None:
        super().__init__(learning_rate)
        self.milestones = milestones
        self.decay = decay
    
    def step(self, epoch: int, n_epochs: int) -> None:
        if epoch in self.milestones:
            self.learning_rate *= self.decay

class StaticScheduler(BaseScheduler):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        super().__init__(learning_rate)
    
    def step(self, epoch: int, n_epochs: int) -> None:
        pass
