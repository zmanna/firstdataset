from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import split_qsar_biodegradation


@dataclass(frozen=True)
class Week7Result:
    model_name: str
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    graph_info: dict[str, int]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _build_descriptor_adjacency(X_train: np.ndarray, top_k: int = 4) -> np.ndarray:
    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    n_nodes = corr.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for node in range(n_nodes):
        neighbors = np.argsort(np.abs(corr[node]))[-top_k:]
        adjacency[node, neighbors] = np.abs(corr[node, neighbors])
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(n_nodes, dtype=np.float64)

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(np.clip(degree, 1e-8, None)))
    return inv_sqrt_degree @ adjacency @ inv_sqrt_degree


class DescriptorGraphNetwork:
    def __init__(
        self,
        adjacency: np.ndarray,
        hidden_dim_1: int = 16,
        hidden_dim_2: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 120,
        random_state: int = 42,
    ) -> None:
        self.adjacency = adjacency
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self._init_params()

    def _init_params(self) -> None:
        rng = np.random.default_rng(self.random_state)
        self.W1 = rng.normal(0.0, 0.15, size=(1, self.hidden_dim_1))
        self.b1 = np.zeros(self.hidden_dim_1)
        self.W2 = rng.normal(0.0, 0.15, size=(self.hidden_dim_1, self.hidden_dim_2))
        self.b2 = np.zeros(self.hidden_dim_2)
        self.Wc = rng.normal(0.0, 0.15, size=(self.hidden_dim_2, 1))
        self.bc = np.zeros(1)

    def _forward(self, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        X0 = sample.reshape(-1, 1)
        AX0 = self.adjacency @ X0
        Z1 = AX0 @ self.W1 + self.b1
        H1 = _relu(Z1)

        AH1 = self.adjacency @ H1
        Z2 = AH1 @ self.W2 + self.b2
        H2 = _relu(Z2)

        graph_embedding = H2.mean(axis=0)
        logit = float(graph_embedding @ self.Wc[:, 0] + self.bc[0])
        prob = float(_sigmoid(logit))
        return AX0, Z1, H1, Z2, H2, prob

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        for _ in range(self.epochs):
            for sample, target in zip(X_train, y_train):
                AX0, Z1, H1, Z2, H2, prob = self._forward(sample)

                target_value = float(target)
                dlogit = prob - target_value

                graph_embedding = H2.mean(axis=0)
                dWc = graph_embedding[:, None] * dlogit
                dbc = np.array([dlogit])

                dgraph = self.Wc[:, 0] * dlogit
                dH2 = np.repeat((dgraph / H2.shape[0])[None, :], H2.shape[0], axis=0)
                dZ2 = dH2 * (Z2 > 0)

                AH1 = self.adjacency @ H1
                dW2 = AH1.T @ dZ2
                db2 = dZ2.sum(axis=0)
                dAH1 = dZ2 @ self.W2.T
                dH1 = self.adjacency.T @ dAH1
                dZ1 = dH1 * (Z1 > 0)

                dW1 = AX0.T @ dZ1
                db1 = dZ1.sum(axis=0)

                self.Wc -= self.learning_rate * dWc
                self.bc -= self.learning_rate * dbc
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.array([self._forward(sample)[-1] for sample in X], dtype=np.float64)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)


def run_week7_descriptor_graph_prototype(random_state: int = 42) -> Week7Result:
    split = split_qsar_biodegradation(random_state=random_state, target_as_category=False)
    y_train = (split.y_train == 2).astype(int).to_numpy()
    y_test = (split.y_test == 2).astype(int).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(split.X_train)
    X_test = scaler.transform(split.X_test)

    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train,
    )

    adjacency = _build_descriptor_adjacency(X_train_inner)
    candidates = [
        DescriptorGraphNetwork(adjacency, learning_rate=0.01, epochs=80, random_state=random_state),
        DescriptorGraphNetwork(adjacency, learning_rate=0.005, epochs=120, random_state=random_state),
    ]

    best_model: DescriptorGraphNetwork | None = None
    best_val_auc = -1.0
    for candidate in candidates:
        candidate.fit(X_train_inner, y_train_inner)
        val_proba = candidate.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = candidate

    assert best_model is not None
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = best_model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=1)),
        "recall": float(recall_score(y_test, predictions, pos_label=1)),
        "f1_score": float(f1_score(y_test, predictions, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    graph_info = {
        "num_nodes": int(adjacency.shape[0]),
        "top_k_neighbors": 4,
    }
    return Week7Result(
        model_name="descriptor_graph_neural_network_prototype",
        metrics=metrics,
        confusion_matrix=matrix,
        graph_info=graph_info,
    )
