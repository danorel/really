from abc import ABC


class AbstractRunner(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if args.optimize:
            self.optimize()
        else:
            self.learn()

    def optimize(self):
        pass

    def learn(self):
        pass
