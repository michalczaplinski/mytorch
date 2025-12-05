from __future__ import annotations

from typing import Iterable
import numpy as np
from .autograd import Tensor


class SGD:
    """
    Implements stochastic gradient descent.
    """

    params: list[Tensor]
    lr: float

    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        """
        Args:
            params: An iterable of Tensors to optimize.
            lr: Learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        """Performs a single optimization step."""
        for p in self.params:
            if p.grad is not None:
                # The core update rule
                p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """
        Clears the gradients of all parameters this optimizer is managing.
        (Convenience method, same as model.zero_grad()).
        """
        for p in self.params:
            p.zero_grad()


class Adam:
    """
    Implements Adam algorithm (Adaptive Moment Estimation).

    Adam combines two powerful ideas:

    1. **Momentum (The Engine)**: Keeps a running average of the gradients (the "First Moment", m).
       This gives the optimizer inertia, allowing it to plow through noisy data and shallow
       local minima while maintaining direction. Like a heavy ball rolling downhill, it
       accumulates velocity and doesn't get derailed by small bumps.

    2. **Adaptive Scaling / RMSProp (The Suspension)**: Keeps a running average of the squared
       gradients (the "Second Moment", v). This scales the learning rate for each parameter
       individually. Parameters with consistently large gradients get smaller effective learning
       rates (preventing overshooting), while parameters with small gradients get larger ones
       (speeding up learning in flat regions).

    3. **Bias Correction**: Since m and v are initialized to zero, they're biased toward zero
       in early training steps. Bias correction mathematically boosts these moving averages
       to compensate, preventing a "slow start" where the optimizer would otherwise take
       tiny steps initially.
    """

    params: list[Tensor]
    lr: float
    beta1: float  # Exponential decay rate for first moment (momentum)
    beta2: float  # Exponential decay rate for second moment (RMSProp)
    eps: float  # Small constant for numerical stability (prevents division by zero)
    state: dict[Tensor, dict[str, np.ndarray]]  # Per-parameter state (m and v)
    t: int  # Time step counter (needed for bias correction)

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {}
        self.t = 0

    def step(self) -> None:
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            # Initialize state for this parameter if it doesn't exist
            # m: First moment (momentum) - exponential moving average of gradients
            # v: Second moment (RMSProp) - exponential moving average of squared gradients
            if p not in self.state:
                self.state[p] = {"m": np.zeros_like(p.data), "v": np.zeros_like(p.data)}

            state = self.state[p]
            m, v = state["m"], state["v"]
            g = p.grad

            # ─────────────────────────────────────────────────────────────────────
            # 1. MOMENTUM (First Moment) - "The Engine"
            # ─────────────────────────────────────────────────────────────────────
            # Exponential moving average of gradients: m = β₁·m + (1-β₁)·g
            # This smooths out noise in gradients and builds up velocity in
            # consistent directions. β₁=0.9 means ~90% of previous momentum is
            # retained, giving the optimizer "memory" of past gradient directions.
            m[:] = self.beta1 * m + (1 - self.beta1) * g

            # ─────────────────────────────────────────────────────────────────────
            # 2. ADAPTIVE SCALING / RMSProp (Second Moment) - "The Suspension"
            # ─────────────────────────────────────────────────────────────────────
            # Exponential moving average of squared gradients: v = β₂·v + (1-β₂)·g²
            # √v approximates the RMS (root mean square) of recent gradients.
            # Dividing by √v normalizes the update: large gradients → smaller steps,
            # small gradients → larger steps. This per-parameter adaptation is key
            # to handling sparse gradients and varying curvature across dimensions.
            v[:] = self.beta2 * v + (1 - self.beta2) * (g**2)

            # ─────────────────────────────────────────────────────────────────────
            # 3. BIAS CORRECTION - "The Warm-Up"
            # ─────────────────────────────────────────────────────────────────────
            # Problem: m and v are initialized to 0, so early estimates are biased
            # toward zero. At t=1 with β₁=0.9: m = 0.1·g (only 10% of the gradient!)
            #
            # Solution: Divide by (1 - βᵗ) which starts near 0 and approaches 1.
            # At t=1: m_hat = m / 0.1 = g (full gradient, as expected)
            # At t=10: m_hat ≈ m / 0.65 (still boosted)
            # At t→∞: m_hat ≈ m (correction fades away as bias disappears)
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            # ─────────────────────────────────────────────────────────────────────
            # 4. PARAMETER UPDATE - Putting it all together
            # ─────────────────────────────────────────────────────────────────────
            # θ = θ - lr · m_hat / (√v_hat + ε)
            #
            # - m_hat provides direction (momentum-smoothed gradient)
            # - √v_hat provides per-parameter scaling (adaptive learning rate)
            # - ε prevents division by zero when v_hat is tiny
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Clears the gradients of all parameters."""
        for p in self.params:
            p.zero_grad()
