from typing import Sequence

import torch
from torch import Tensor
from torch.distributions import Binomial


class BinomialGE(torch.nn.Module):
    def __init__(
        self, pi: float = None, slack: float = 1.0, entropy_penalty: float = 0.0
    ) -> None:
        super(BinomialGE, self).__init__()
        self.pi = pi
        self.slack = slack
        self.entropy_penalty = entropy_penalty
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        select = targets.data == 1
        classifier_loss = self.criterion(outputs[select], targets[select])
        # 1. Calculate Normal approximation to the distribution over positive count given by
        # the classifier
        select = targets.data == 0
        N = select.sum().item()
        p_hat = outputs[select].sigmoid()
        q_mu = p_hat.sum()
        q_var = torch.sum(p_hat * (1 - p_hat))
        count_vector = torch.arange(
            0, N + 1, device=targets.device, dtype=torch.float32
        )
        # below, 1e-10 is only a small epsilon to prevent NaN
        q_discrete = (-0.5 * (q_mu - count_vector) ** 2 / (q_var + 1e-10)).softmax(
            dim=0
        )
        # 2. KL of w from the binomial distribution with pi
        log_binom = Binomial(total_count=N, probs=self.pi).log_prob(count_vector)
        ge_penalty = -torch.sum(log_binom * q_discrete)
        if self.entropy_penalty > 0:
            q_entropy = 0.5 * (torch.log(q_var) + torch.log(2 * self.pi) + 1)
            ge_penalty += q_entropy * self.entropy_penalty
        loss = classifier_loss + self.slack * ge_penalty
        return loss


class MultiTaskLoss(torch.nn.Module):
    def __init__(
        self,
        criteria: Sequence[torch.nn.Module],
        eta: Sequence[float] = [0],
        learn_eta: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """A module to compute a loss consisting of several terms, whose weights can be learnt.

        Args:
            criteria (Tuple[torch.nn.Module]): Tuple of losses, i.e. each terms constituting the
                                               total loss.
            eta (List[float], optional): Weighting of each losses. This MUST be equal in length
                                         to criteria. Defaults to [0].
            learn_eta (bool, optional): If False, weightings will remain constants during
                                        training. Otherwhise, they'll be optimized.
                                        Defaults to False.
            device (torch.device, optional): A torch device, typically cpu or gpu.
                                             Defaults to torch.device('cpu').
        """
        super(MultiTaskLoss, self).__init__()
        self.criteria = criteria
        self.eta = torch.Tensor(eta).to(device)
        if learn_eta:
            self.eta = torch.nn.Parameter(self.eta)

    def forward(
        self, outputs: Sequence[Tensor], targets: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        loss = [
            criterion(o, y) for criterion, o, y in zip(self.criteria, outputs, targets)
        ]
        total_loss = torch.stack(loss) * torch.exp(-self.eta) + self.eta
        return loss, total_loss.sum()
