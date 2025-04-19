import torch
from geomloss import SamplesLoss


def compute_pairwise_2_wasserstein(posteriors, p=2, blur=0.05):
    """
    Compute an NxN matrix of pairwise 2-Wasserstein distances
    between the posterior distributions of N assets.

    Args:
        posteriors (list[torch.Tensor] or torch.Tensor):
            Each element is a (M, d) Tensor of posterior samples
            (M samples, d-dimensional).
        p (float): Power used in the Sinkhorn distance (p=2 for 2-Wasserstein).
        blur (float): Regularization parameter for the Sinkhorn algorithm.

    Returns:
        torch.Tensor: An (N x N) matrix containing the 2-Wasserstein distances.
    """

    # If posteriors is a single Tensor of shape (N, M, d), convert it to a list.
    # Otherwise, we assume it is already a list of length N, each (M, d).
    if isinstance(posteriors, torch.Tensor):
        N = posteriors.shape[0]
        posteriors_list = [posteriors[i] for i in range(N)]
    else:
        posteriors_list = posteriors
        N = len(posteriors_list)

    # Move data to GPU if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    posteriors_list = [posterior.to(device) for posterior in posteriors_list]

    # Create the SamplesLoss object for 2-Wasserstein
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    # Allocate an NxN distance matrix
    dist_matrix = torch.zeros((N, N), device=device)

    # Compute pairwise distances
    for i in range(N):
        for j in range(i, N):
            if i == j:
                dist = 0.0
            else:
                dist = sinkhorn_loss(posteriors_list[i], posteriors_list[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # symmetry

    # Optionally move the result back to CPU
    return dist_matrix.cpu()


def compute_pairwise_2_wasserstein_batched(posteriors, p=2, blur=0.05, chunk_size=256):
    """
    Compute an NxN matrix of pairwise 2-Wasserstein distances between
    posterior distributions of N assets, batching the distance calls
    for faster GPU usage. This version uses a single GPU but batches
    multiple pairs into a single Sinkhorn call.

    Args:
        posteriors (list[torch.Tensor] or torch.Tensor):
            Each element is a (M, d) Tensor of posterior samples
            (M samples, d-dimensional).
            Alternatively, a single Tensor of shape (N, M, d).
        p (float): Power used in the Sinkhorn distance (2 for 2-Wasserstein).
        blur (float): Regularization parameter for the Sinkhorn algorithm.
        chunk_size (int): Number of pairs to process at once.

    Returns:
        torch.Tensor: An (N x N) matrix of 2-Wasserstein distances.
    """
    # Convert input to a list of (M, d) Tensors if needed
    if isinstance(posteriors, torch.Tensor):
        N = posteriors.shape[0]
        posteriors_list = [posteriors[i] for i in range(N)]
    else:
        posteriors_list = posteriors
        N = len(posteriors_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    posteriors_list = [posterior.to(device) for posterior in posteriors_list]

    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    # Pre-allocate NxN distance matrix on GPU
    dist_matrix = torch.zeros((N, N), device=device)

    # Gather all upper-triangular pairs (i < j)
    # and skip i==j, since distance is zero
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))

    # Chunk the pairs to avoid GPU memory blowups
    # and reduce overhead from repeated calls.
    start = 0
    while start < len(pairs):
        end = min(start + chunk_size, len(pairs))
        current_pairs = pairs[start:end]

        # Build xBatch and yBatch for the chunk
        x_batch = []
        y_batch = []
        for i, j in current_pairs:
            x_batch.append(posteriors_list[i])
            y_batch.append(posteriors_list[j])
        # shape of each is [B, M, d]
        x_batch = torch.stack(x_batch, dim=0)
        y_batch = torch.stack(y_batch, dim=0)

        # Compute distances for the entire chunk at once
        # This returns a 1D tensor of size B
        dists = sinkhorn_loss(x_batch, y_batch)

        # Fill the dist_matrix (upper + lower)
        for pair_idx, (i, j) in enumerate(current_pairs):
            dist_matrix[i, j] = dists[pair_idx]
            dist_matrix[j, i] = dists[pair_idx]

        start = end  # move to next chunk

    # Zero out diagonal, though it should already be zero
    # (or you can skip this if it's obviously zero already).
    dist_matrix.fill_diagonal_(0.0)

    return dist_matrix.cpu()
