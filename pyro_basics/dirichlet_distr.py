import torch
import pyro
import pyro.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt


alpha_list = [
    torch.tensor([0.9, 0.1]),
    torch.tensor([2.0, 2.0]),
    torch.tensor([2.0, 5.0]),
    torch.tensor([5.0, 2.0]),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, alpha in enumerate(alpha_list):
    row, col = divmod(idx, 2)

    # Build the Dirichlet distribution object.
    dirichlet_dist = dist.Dirichlet(alpha)

    # Draw samples; each sample is a 2D vector that sums to 1.
    samples = dirichlet_dist.sample(sample_shape=(10_000,))
    first_coordinate = samples[:, 0].numpy()

    # Plot the distribution of the first coordinate on a subplot.
    sns.histplot(first_coordinate, bins=50, stat="density", kde=True, alpha=0.4, ax=axes[row][col])

    # Set the title to show the alpha values.
    axes[row][col].set_title(f"Dirichlet alpha={alpha.numpy().tolist()}")
# Adjust spacing and show the figure.

fig.suptitle("Different Dirichlet Distributions (dim=2) in a Grid", fontsize=16)
plt.tight_layout()
plt.show()
