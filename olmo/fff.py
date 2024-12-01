import torch
import torch.nn as nn

import math

class FFF(nn.Module):
	def __init__(self, input_width, output_width, depth, parallel_size, activation=nn.GELU, device=None):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth
		self.parallel_size = parallel_size
		self.n_nodes = 2 ** (self.depth + 1) - 1

		self.linear_in = nn.Linear(input_width, parallel_size * self.n_nodes, bias=True, device=device)
		self.linear_out = nn.Linear(parallel_size * self.n_nodes, output_width, bias=False, device=device)

		init_k = math.sqrt(1.0 / self.input_width)
		self.linear_in.weight.data = torch.empty((self.parallel_size * self.n_nodes, self.input_width), device=device).uniform_(-init_k, +init_k)
		self.linear_in.bias.data = torch.empty((self.parallel_size * self.n_nodes), device=device).uniform_(-init_k, +init_k)
		init_k2 = math.sqrt(1.0 / ((self.depth+1) * self.parallel_size))
		self.linear_out.weight.data = torch.empty((self.output_width, self.parallel_size * self.n_nodes), device=device).uniform_(-init_k2, +init_k2)

		# self.activation = activation()
		self.activation = activation

	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
		# x has shape (..., input_width)
		x = oldx.reshape(-1, self.input_width)
		# x has shape (batch_size, input_width)
		batch_size = x.shape[0]

		logits = self.linear_in(x) # (batch_size, parallel_size * n_nodes)
		logit_decisions = (logits > 0).long() # (batch_size, parallel_size * n_nodes)
		activations = self.activation(logits) # (batch_size, parallel_size * n_nodes)

		# recursively descending by depth, enforce conditionality
		activations = activations.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)
		decisions = logit_decisions.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)

		with torch.no_grad():
			current_nodes = torch.zeros((batch_size, self.parallel_size), dtype=torch.long, device=x.device)
			decision_map = torch.zeros_like(decisions, dtype=torch.float, device=x.device) # (batch_size, parallel_size, n_nodes)
			decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0)

			for d in range(self.depth):
				current_platform = 2 ** d - 1
				next_platform = 2 ** (d + 1) - 1
				moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)
				next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform
				decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)
				current_nodes = next_nodes

		activations = activations * decision_map # (batch_size, parallel_size, n_nodes)
		new_logits = self.linear_out(activations.flatten(1, 2)) # (batch_size, output_width)

		ret = new_logits.reshape_as(oldx)
		return ret
