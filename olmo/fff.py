# import torch
# import torch.nn as nn

# import math

# class FFF(nn.Module):
# 	def __init__(self, input_width, output_width, depth, parallel_size, activation=nn.GELU, device=None):
# 		super().__init__()

# 		self.input_width = input_width
# 		self.output_width = output_width
# 		self.depth = depth
# 		self.parallel_size = parallel_size
# 		self.n_nodes = 2 ** (self.depth + 1) - 1

# 		self.linear_in = nn.Linear(input_width, parallel_size * self.n_nodes, bias=True, device=device)
# 		self.linear_out = nn.Linear(parallel_size * self.n_nodes, output_width, bias=False, device=device)

# 		init_k = math.sqrt(1.0 / self.input_width)
# 		self.linear_in.weight.data = torch.empty((self.parallel_size * self.n_nodes, self.input_width), device=device).uniform_(-init_k, +init_k)
# 		self.linear_in.bias.data = torch.empty((self.parallel_size * self.n_nodes), device=device).uniform_(-init_k, +init_k)
# 		init_k2 = math.sqrt(1.0 / ((self.depth+1) * self.parallel_size))
# 		self.linear_out.weight.data = torch.empty((self.output_width, self.parallel_size * self.n_nodes), device=device).uniform_(-init_k2, +init_k2)

# 		# self.activation = activation()
# 		self.activation = activation

# 	def forward(self, oldx: torch.Tensor) -> torch.Tensor:
# 		# x has shape (..., input_width)
# 		x = oldx.reshape(-1, self.input_width)
# 		# x has shape (batch_size, input_width)
# 		batch_size = x.shape[0]

# 		logits = self.linear_in(x) # (batch_size, parallel_size * n_nodes)
# 		logit_decisions = (logits > 0).long() # (batch_size, parallel_size * n_nodes)
# 		activations = self.activation(logits) # (batch_size, parallel_size * n_nodes)

# 		# recursively descending by depth, enforce conditionality
# 		activations = activations.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)
# 		decisions = logit_decisions.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)

# 		with torch.no_grad():
# 			current_nodes = torch.zeros((batch_size, self.parallel_size), dtype=torch.long, device=x.device)
# 			decision_map = torch.zeros_like(decisions, dtype=torch.float, device=x.device) # (batch_size, parallel_size, n_nodes)
# 			decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0)

# 			for d in range(self.depth):
# 				current_platform = 2 ** d - 1
# 				next_platform = 2 ** (d + 1) - 1
# 				moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)
# 				next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform
# 				decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)
# 				current_nodes = next_nodes

# 		activations = activations * decision_map # (batch_size, parallel_size, n_nodes)
# 		new_logits = self.linear_out(activations.flatten(1, 2)) # (batch_size, output_width)

# 		ret = new_logits.reshape_as(oldx)
# 		return ret


# import torch
# from torch import nn

# class FFF(nn.Module):
# 	def __init__(self, input_width: int, output_width: int, depth: int, activation=nn.GELU, device=None):
# 		super().__init__()

# 		self.input_width = input_width
# 		self.output_width = output_width

# 		self.depth = depth
# 		self.n_nodes = 2 ** (depth + 1) - 1

# 		self.activation = activation

# 		self.linear_in = nn.Parameter(torch.empty(self.n_nodes, self.input_width, device=device), requires_grad=True)
# 		self.linear_out = nn.Parameter(torch.empty(self.n_nodes, self.output_width, device=device), requires_grad=True)

# 	def forward(self, x):
# 		# the shape of x is (batch_size, input_width)
# 		# retrieve the indices of the relevant elements
# 		batch_size = x.shape[0]
# 		current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
# 		all_nodes = torch.zeros(batch_size, self.depth+1, dtype=torch.long, device=x.device)
# 		all_logits = torch.empty((batch_size, self.depth+1), dtype=torch.float, device=x.device)

# 		for i in range(self.depth+1):
# 			all_nodes[:, i] = current_nodes
# 			plane_coeffs = self.linear_in.index_select(dim=0, index=current_nodes)			# (batch_size, input_width)
# 			plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))	# (batch_size, 1, 1)
# 			plane_score = plane_coeff_score.squeeze(-1).squeeze(-1) 					# (batch_size,)
# 			all_logits[:, i] = plane_score
# 			plane_choices = (plane_score >= 0).long()									# (batch_size,)

# 			current_nodes = current_nodes * 2 + plane_choices + 1						# (batch_size,)

# 		# get the weights
# 		selected_linear_out = self.linear_out.index_select(0, index=all_nodes.flatten()) \
# 			.view(batch_size, self.depth+1, self.output_width)	# (batch_size, depth+1, output_width)

# 		# forward pass
# 		# mlp1 = torch.nn.functional.gelu(all_logits)				# (batch_size, depth+1)
# 		mlp1 = self.activation(all_logits)				# (batch_size, depth+1)
# 		mlp2 = torch.bmm(mlp1.unsqueeze(1), selected_linear_out) 		# (batch_size, output_width)
        
# 		# done
# 		return mlp2
    

import torch
import torch.nn as nn

class FFF(nn.Module):
    def __init__(self, input_width: int, output_width: int, depth: int, activation=nn.GELU, device=None):
        super().__init__()
        
        self.input_width = input_width
        self.output_width = output_width
        self.depth = depth
        self.n_nodes = 2 ** (depth + 1) - 1
        
        self.activation = activation

        self.linear_in = nn.Linear(input_width, self.n_nodes, bias=False).to(device)
        self.linear_out = nn.Linear(self.n_nodes, output_width, bias=False).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        all_nodes = torch.zeros(batch_size, self.depth + 1, dtype=torch.long, device=x.device)
        all_logits = torch.empty((batch_size, self.depth + 1), dtype=torch.float, device=x.device)

        for i in range(self.depth + 1):
            all_nodes[:, i] = current_nodes
            plane_score = self.linear_in(x).gather(1, current_nodes.unsqueeze(1)).squeeze(1)
            all_logits[:, i] = plane_score
            plane_choices = (plane_score >= 0).long()
            current_nodes = current_nodes * 2 + plane_choices + 1

        selected_logits = all_logits.flatten(start_dim=0)
        mlp1 = self.activation(selected_logits.view(batch_size, self.depth + 1))
        mlp2 = self.linear_out(mlp1)
        
        return mlp2