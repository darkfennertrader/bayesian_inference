import torch
import pyro.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt


# Define six (alpha, beta) pairs to illustrate different Beta shapes
alpha_beta_values = [
    (0.1, 0.9),  # Skewed heavily toward 0
    (1.0, 1.0),  # Uniform
    (2.0, 2.0),  # Near the middle
    (2.0, 5.0),  # Skewed toward 0
    (5.0, 2.0),  # Skewed toward 1
    (10.0, 10.0),  # More concentrated around 0.5
]


x = torch.linspace(0, 1, 200)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, (alpha_val, beta_val) in enumerate(alpha_beta_values):
    # Create Beta distribution with Pyro
    beta_dist = dist.Beta(torch.tensor(alpha_val), torch.tensor(beta_val))
    # Compute PDF by exponentiating the log_prob
    pdf = beta_dist.log_prob(x).exp()

    axes[i].plot(x.numpy(), pdf.numpy())
    axes[i].set_title(f"alpha={alpha_val}, beta={beta_val}")
    axes[i].set_xlabel("probability")
    axes[i].set_ylabel("pdf")

plt.tight_layout()
plt.show()
