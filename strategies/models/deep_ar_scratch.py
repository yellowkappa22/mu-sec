import torch
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.normal import Normal
from typing import Tuple
from pprint import pprint
import math

class DeepAR(torch.nn.Module):
    def __init__(self, hidden_size: int, x_size: int, cont_mode=True, lr=0.01, momentum=0.9):
        super().__init__()
        self.params = torch.nn.ParameterDict(
            {
                # forget
                "Wf": torch.nn.Parameter(torch.empty(hidden_size, x_size + 1)),
                "Uf": torch.nn.Parameter(torch.empty(hidden_size, hidden_size)),
                "bf": torch.nn.Parameter(torch.zeros(hidden_size,)),
                # input
                "Wi": torch.nn.Parameter(torch.empty(hidden_size, x_size + 1)),
                "Ui": torch.nn.Parameter(torch.empty(hidden_size, hidden_size)),
                "bi": torch.nn.Parameter(torch.zeros(hidden_size,)),
                # candidate
                "Wc": torch.nn.Parameter(torch.empty(hidden_size, x_size + 1)),
                "Uc": torch.nn.Parameter(torch.empty(hidden_size, hidden_size)),
                "bc": torch.nn.Parameter(torch.zeros(hidden_size,)),
                # output
                "Wo": torch.nn.Parameter(torch.empty(hidden_size, x_size + 1)),
                "Uo": torch.nn.Parameter(torch.empty(hidden_size, hidden_size)),
                "bo": torch.nn.Parameter(torch.zeros(hidden_size,)),
                # probability
                "W1": torch.nn.Parameter(torch.empty(1, hidden_size)),
                "b1": torch.nn.Parameter(torch.zeros(1,)),
                "W2": torch.nn.Parameter(torch.empty(1, hidden_size)),
                "b2": torch.nn.Parameter(torch.zeros(1,))
            }
        )
        for w, p in self.params.items():
            if w[0] != "b":
                torch.nn.init.xavier_uniform_(p)

        self.cont_mode = cont_mode
        self.hidden_size = hidden_size
        self.optim = torch.optim.SGD(self.params.parameters(), lr=lr, momentum=momentum)
    
    def gaussian_output_layer(self, hidden_state: torch.Tensor, eps=1e-6):
        mu = hidden_state @ self.params["W1"].T + self.params["b1"]
        sigma = torch.nn.functional.softplus(
            hidden_state @ self.params["W2"].T + self.params["b2"]
        )
        return mu, sigma + eps
    
    def gaussian_neg_log_likelihood(self, z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        neg_log_lh = (
            torch.log(sigma) 
            + 0.5 * ((z - mu) / sigma)**2
        )
        return neg_log_lh
    
    def negbin_output_layer(self, hidden_state: torch.Tensor, eps=1e-6):
        mu = torch.nn.functional.softplus(
            hidden_state @ self.params["W1"].T + self.params["b1"]
        )
        alpha = torch.nn.functional.softplus(
            hidden_state @ self.params["W2"].T + self.params["b2"]
        )
        return mu + eps, alpha + eps 
    
    def negbin_neg_log_likelihood(self, z: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor):
        r = 1.0 / alpha
        log1p_prod = torch.log1p(alpha * mu)
        neg_log_lh = (
                - torch.lgamma(z + r)
                + torch.lgamma(z + 1.0)
                + torch.lgamma(r)
                + r * log1p_prod
                - torch.xlogy(z, torch.log(alpha * mu)) + z * log1p_prod
        )
        return neg_log_lh
        
    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zx = torch.cat((z, x), dim=1) # [b (dim=0), f (dim=1)], [[x111, x112, x113, y11], [x121, x122, x123, y12], [x131, x132, x133, y13]] (t_id, b_id, f_id)
        f = torch.sigmoid(zx @ self.params["Wf"].T + hidden_state @ self.params["Uf"].T + self.params["bf"])
        i = torch.sigmoid(zx @ self.params["Wi"].T + hidden_state @ self.params["Ui"].T + self.params["bi"])
        c = torch.tanh(zx @ self.params["Wc"].T + hidden_state @ self.params["Uc"].T + self.params["bc"])
        o = torch.sigmoid(zx @ self.params["Wo"].T + hidden_state @ self.params["Uo"].T + self.params["bo"])
        
        cell_state = f * cell_state + i * c
        hidden_state = o * torch.tanh(cell_state)
        
        return hidden_state, cell_state
    
    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        try:
            nT, nB, nF = X.shape
        except Exception as exc:
            raise ValueError(f"\nincorrect 'X' dimensions\n{X.shape}\n{exc}\n")

        if self.cont_mode:
            output_layer = self.gaussian_output_layer
            neg_log_lh_func = self.gaussian_neg_log_likelihood
        else:
            output_layer = self.negbin_output_layer
            neg_log_lh_func = self.negbin_neg_log_likelihood
            
        total_neg_log_lh = torch.zeros(nB, 1, device=X.device)
        hidden_state = torch.zeros(nB, self.hidden_size, device=X.device)
        cell_state = torch.zeros(nB, self.hidden_size, device=X.device)
        z = torch.zeros(nB, 1, device=X.device)
        for t in range(nT):
            x = X[t]
            y = Y[t].unsqueeze(1)
            hidden_state, cell_state = self.forward(z, x, hidden_state, cell_state)
            total_neg_log_lh += neg_log_lh_func(y, *output_layer(hidden_state))
            z = y
        
        self.optim.zero_grad()
        loss = total_neg_log_lh.mean()
        loss.backward()
        self.optim.step()

cell = DeepAR(64, 32)
pprint(cell.params)