class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        EarlyStopping class to monitor the validation loss and stop if no improvement for 'patience' epochs.

        Args:
            patience (int): How many epochs to wait after the last time validation loss improved.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        """
        Call method to check if the validation loss improved. If not, increase the counter.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        score = -val_loss  # We want to minimize the validation loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def get_best_weights(self):
        """
        Returns the weights of the best model found so far.
        """
        return self.best_model_wts
