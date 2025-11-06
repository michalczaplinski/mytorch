class SGD:
    """
    Implements stochastic gradient descent.
    """
    def __init__(self, params, lr=0.01):
        """
        Args:
            params (iterable): An iterable of Tensors to optimize.
            lr (float): Learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Performs a single optimization step."""
        for p in self.params:
            if p.grad is not None:
                # The core update rule
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Clears the gradients of all parameters this optimizer is managing.
        (Convenience method, same as model.zero_grad()).
        """
        for p in self.params:
            p.zero_grad()