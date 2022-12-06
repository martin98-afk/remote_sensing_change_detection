import matplotlib.pyplot as plt


class AverageMeter():
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.avg_history = []

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.val = val
        self.avg = self.sum / self.count
        self.avg_history.append(self.avg)

    def plot(self, name):
        plt.figure(figsize=(8, 8))
        plt.plot(self.avg_history)
        plt.grid('on')
        plt.title(f'{name} curve')
