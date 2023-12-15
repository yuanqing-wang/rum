class EarlyStopping(object):
    """Early stopping.

    Parameters
    ----------
    patience : int = 10
        Patience for early stopping.

    """

    best_losses = None
    counter = 0

    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, losses):
        if self.best_losses is None:
            self.best_losses = losses
            self.counter = 0

        elif any(
            loss <= best_loss
            for loss, best_loss in zip(losses, self.best_losses)
        ):
            if all(
                loss <= best_loss
                for loss, best_loss in zip(losses, self.best_losses)
            ):

                self.best_losses = [
                    min(loss, best_loss)
                    for loss, best_loss in zip(losses, self.best_losses)
                ]
                self.counter = 0

        else:
            self.counter += 1
            if self.counter == self.patience:
                return True

        return False