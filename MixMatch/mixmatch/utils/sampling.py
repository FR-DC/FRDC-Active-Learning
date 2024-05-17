import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple
from kmeans_pytorch import kmeans

def shuffled_argmin(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Shuffles the values and sorts them afterwards. This can be used to break
    the tie when the highest utility score is not unique. The shuffle randomizes
    order, which is preserved by the mergesort algorithm.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices and values to return.
    Returns:
        The indices and values of the n_instances smallest values.
    """

    indexes, index_values = shuffled_argmax(-values, n_instances)

    return indexes, -index_values

def shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> torch.Tensor:
    """
    Shuffles the values and sorts them afterwards. This can be used to break
    the tie when the highest utility score is not unique. The shuffle randomizes
    order, which is preserved by the mergesort algorithm.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices and values to return.
    Returns:
        The indices and values of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    # shuffling indices and corresponding values
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]

    # getting the n_instances best instance
    # since mergesort is used, the shuffled order is preserved
    sorted_query_idx = np.argsort(shuffled_values, kind='mergesort')[
        len(shuffled_values)-n_instances:]

    # inverting the shuffle
    query_idx = shuffled_idx[sorted_query_idx]

    return query_idx, values[query_idx]

def multi_argmax(values: torch.Tensor, n_instances: int = 1) -> Tuple[torch.Tensor, float]:
    """
    return the indices and values of the n_instances highest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices and values to return.
    Returns:
        The indices and greatest uncertainty score.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'
    top_k = torch.topk(values, k=n_instances)
    return top_k.indices, torch.max(top_k.values).to("cpu").item()


def multi_argmin(values: torch.Tensor, n_instances: int = 1) -> Tuple[torch.Tensor, float]:
    """
    return the indices and values of the n_instances smallest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices and values to return.
    Returns:
        The indices and smallest uncertainty score.
    """
    indexes, smallest = multi_argmax(-values, n_instances)
    return indexes, -smallest

def classifier_confidence(model: nn.Module, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(X)
    softmax = nn.Softmax(dim=1)
    preds = softmax(outputs)
    preds1 = model.final_hidden(X)
    # for each point, select the maximum uncertainty
    uncertainty = 1 - (torch.max(preds, dim=1).values)
    return uncertainty, preds1

def classifier_margin(model: nn.Module, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(X)
    softmax = nn.Softmax(dim=1)
    preds = softmax(outputs)
    preds1 = model.final_hidden(X)
    # for each point, select the top 2 most confident scores
    top_2 = torch.topk(preds, 2, axis=1)[0]
    return top_2[:, 0] - top_2[:, 1], preds1

def classifier_entropy(model: nn.Module, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(X)
    softmax = nn.Softmax(dim=1)
    preds = softmax(outputs)
    preds1 = model.final_hidden(X)
    entropy = Categorical(probs = preds).entropy()
    return entropy, preds1

def confidence_sampling(model: nn.Module, 
                         X: torch.Tensor,
                         n_instances: int,
                         random_tie_break: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    confidence, preds = classifier_confidence(model, X)

    if not random_tie_break:
        max_idxes, max_uncertainty = multi_argmax(confidence, n_instances=n_instances)
        X_max = X[max_idxes]
        max_idxes = max_idxes.to("cpu").numpy()
        X_max = X_max.permute(0, 2, 3, 1).to("cpu").numpy()
        return preds.to("cpu").detach().numpy(), max_idxes, X_max, max_uncertainty

    return preds.to("cpu").detach().numpy(), shuffled_argmax(confidence, n_instances=n_instances)

def margin_sampling(model: nn.Module,
                    X: torch.Tensor,
                    n_instances: int,
                    random_tie_break: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    margin, preds = classifier_margin(model, X)

    if not random_tie_break:
        min_idxes, min_margins = multi_argmin(margin, n_instances=n_instances)
        X_min = X[min_idxes]
        min_idxes = min_idxes.to("cpu").numpy()
        X_min = X_min.permute(0, 2, 3, 1).to("cpu").numpy()
        return preds.to("cpu").detach().numpy(), min_idxes, X_min, min_margins

    return preds.to("cpu").detach().numpy(), shuffled_argmin(margin, n_instances=n_instances)

def entropy_sampling(model: nn.Module,
                     X: torch.Tensor,
                     n_instances: int,
                     random_tie_break: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    entropy, preds = classifier_entropy(model, X)

    if not random_tie_break:
        max_idxes, max_entropy = multi_argmax(entropy, n_instances=n_instances)
        X_max = X[max_idxes]
        max_idxes = max_idxes.to("cpu").numpy()
        X_max = X_max.permute(0, 2, 3, 1).to("cpu").numpy()
        return preds.to("cpu").detach().numpy(), max_idxes, X_max, max_entropy

    return preds.to("cpu").detach().numpy(), shuffled_argmax(entropy, n_instances=n_instances)

def random_sampling(model: nn.Module,
                    X: torch.Tensor,
                    n_instances: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    outputs = model(X)
    softmax = nn.Softmax(dim=1)
    preds = softmax(outputs).to("cpu").detach().numpy()
    # preds = model.final_hidden(X).to("cpu").detach().numpy()
    perm = torch.randperm(X.size(0))
    random_idxes = perm[:n_instances].to("cpu").numpy()
    random_samples = X[random_idxes].permute(0, 2, 3, 1).to("cpu").numpy()
    return preds, random_idxes, random_samples, 0.5

def cluster_sampling(model: nn.Module,
                     X: torch.Tensor,
                     num_clusters: int,
                     n_instances: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    feats = model.final_hidden(X).to("cpu")
    cluster_ids_x, cluster_centers = kmeans(
        X=feats, num_clusters=num_clusters, distance='cosine', device="cuda"
    )
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    num_per_cluster = n_instances // num_clusters
    idxes, samples = [], []
    avg = 0.0
    for i in range(num_clusters):
        feat_idxes = torch.where(cluster_ids_x == i)[0]
        feat_vecs = feats[feat_idxes]
        cos_distances = cos(feat_vecs, cluster_centers[i])
        sorted_list = sorted(list(zip(feat_idxes, feat_vecs, cos_distances)), key=lambda x:x[2])
        idx, _, s_distances = list(zip(*sorted_list))
        avg += s_distances[-1] - s_distances[0]
        idxes.extend([idx[:num_per_cluster//2]])
        idxes.extend([idx[-num_per_cluster//2:]])
        samples.extend([X[:num_per_cluster//2].permute(0, 2, 3, 1).to("cpu").numpy()])
        samples.extend([X[-num_per_cluster//2:].permute(0, 2, 3, 1).to("cpu").numpy()])
    idxes = np.array(idxes).squeeze()
    samples = np.array(samples).squeeze()
    return feats.detach().numpy(),idxes, samples, avg.item()/num_clusters

def model_outlier_sampling(model: nn.Module, 
                           test_data: torch.Tensor,
                           validation_data: torch.Tensor,
                           n_instances: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    val_logits = model.final_hidden(validation_data)
    max_rank = torch.max(val_logits, dim=0).values.unsqueeze(0)
    min_rank = torch.min(val_logits, dim=0).values.unsqueeze(0)
    test_logits = model.final_hidden(test_data)
    ranks = (test_logits - min_rank) / (max_rank - min_rank)
    mean_scores = torch.mean(torch.clamp(ranks, 0, 1), dim=1)
    s = [(i, score) for i, score in enumerate(mean_scores)]
    s.sort(key=lambda x:x[1])
    idxes = []
    avg = 0.0
    for i in range(n_instances):
        idxes.append(s[i][0])
        avg += s[i][1].item()
    return (test_logits.to("cpu").detach().numpy(),
            np.array(idxes), 
            test_data[idxes].permute(0, 2, 3, 1).to("cpu").numpy(), 
            avg/n_instances)

def hybrid_sampling(model: nn.Module,
                    validation_data: torch.Tensor,
                    test_data: torch.Tensor,
                    n_instances: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Hybrid of random, entropy and cluster sampling 
    """
    feats = model.final_hidden(test_data).to("cpu").detach().numpy()
    _, outlier_idxes, outlier_samples, outlier_scores = model_outlier_sampling(model, validation_data, test_data, n_instances=2)
    _, entropy_idxes, entropy_samples, entropy_scores = entropy_sampling(model, test_data, n_instances=2)
    _, random_idxes, random_samples, _ = random_sampling(model, test_data, n_instances=16)
    return (feats,
            np.concatenate([outlier_idxes, entropy_idxes, random_idxes]), 
            np.concatenate([outlier_samples, entropy_samples, random_samples]), 
            outlier_scores + entropy_scores)



    