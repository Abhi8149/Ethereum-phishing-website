from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import GraphNorm, PDNConv, SAGEConv


DAHGNN_WEIGHTS_FILE = "best_dahgnn_model.pt"
GAE_WEIGHTS_FILE = "gae_pdna_weights.pth"
RF_MODEL_FILE = "random_forest_model.pkl"
ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"


@dataclass
class PredictionResult:
    address: str
    dahgnn_probability: float
    dahgnn_label: str
    gae_rf_probability: float
    gae_rf_label: str
    final_probability: float
    final_label: str


@dataclass
class WalletTransactionRecord:
    address: str
    summary: Dict[str, object]
    transactions: pd.DataFrame


@dataclass
class InferenceContext:
    dahgnn_feature_vector: np.ndarray
    dahgnn_node_features: torch.Tensor
    gae_node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_idx: int


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GRUWithMHA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.gru1(x)
        attn_out, _ = self.mha(out1, out1, out1)
        out2, _ = self.gru2(attn_out)
        return out2


class TemporalFeaturesModule(nn.Module):
    def __init__(self, num_features: int, conv_channels: int = 32, gru_hidden: int = 32):
        super().__init__()
        self.conv_branch = Conv1DBlock(in_channels=num_features, out_channels=conv_channels)
        self.gru_mha_branch = GRUWithMHA(input_size=num_features, hidden_size=gru_hidden)
        self.out_dim = conv_channels + gru_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_in = x.permute(0, 2, 1)
        conv_out = self.conv_branch(conv_in)
        conv_pool = conv_out.mean(dim=2)

        gru_out = self.gru_mha_branch(x)
        gru_pool = gru_out.mean(dim=1)
        return torch.cat([conv_pool, gru_pool], dim=1)


class SAGEConvReconstructionModule(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.1)
        z = F.leaky_relu(self.conv2(h, edge_index), negative_slope=0.1)
        return z


class DAHGNN(nn.Module):
    def __init__(self, num_basic_features: int, temporal_seq_features: int, temporal_window: int,
                 graph_hidden_dim: int = 64, graph_out_dim: int = 32):
        super().__init__()
        self.temporal = TemporalFeaturesModule(
            num_features=temporal_seq_features,
            conv_channels=32,
            gru_hidden=32,
        )
        self.sage = SAGEConvReconstructionModule(
            in_dim=num_basic_features,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
        )
        combined_dim = num_basic_features + self.temporal.out_dim + graph_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.temporal_window = temporal_window

    def forward(
        self,
        time_series_input: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        basic_features: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        temporal_feats = self.temporal(time_series_input)
        graph_feats_all = self.sage(node_features, edge_index)
        graph_feats = graph_feats_all[node_idx]
        combined = torch.cat([basic_features, temporal_feats, graph_feats], dim=1)
        return self.classifier(combined).squeeze(-1)


class PDNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, edge_hidden: int):
        super().__init__()
        self.conv = PDNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            hidden_channels=edge_hidden,
        )
        self.act = nn.PReLU()
        self.norm = GraphNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.norm(x)
        return x


class GAE_PDNA_Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 edge_dim: int, edge_hidden: int, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(PDNBlock(in_channels, hidden_channels, edge_dim, edge_hidden))
        for _ in range(num_blocks - 1):
            self.blocks.append(PDNBlock(hidden_channels, hidden_channels, edge_dim, edge_hidden))
        self.final_conv = PDNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            hidden_channels=edge_hidden,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        x = self.final_conv(x, edge_index, edge_attr)
        return x


class InnerProductDecoder(nn.Module):
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)


class GAE_PDNA(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 15,
                 edge_dim: int = 3, edge_hidden: int = 6, num_blocks: int = 4):
        super().__init__()
        self.encoder = GAE_PDNA_Encoder(
            in_channels, hidden_channels, out_channels, edge_dim, edge_hidden, num_blocks
        )
        self.decoder = InnerProductDecoder()

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.encode(x, edge_index, edge_attr)


def _normalize_address(address: str) -> str:
    return str(address).strip().lower()


def _to_float(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _minmax_scale(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    return ((matrix - mins) / denom).astype(np.float32)


def _build_temporal_matrix_from_features(feature_vector: np.ndarray) -> np.ndarray:
    num_features = feature_vector.shape[0]
    return np.stack([np.roll(feature_vector, shift=i) for i in range(num_features)], axis=0)


def make_node_windows(feature_vector: np.ndarray, window_size: int = 4, stride: int = 2) -> np.ndarray:
    sequence = _build_temporal_matrix_from_features(feature_vector)

    if sequence.shape[0] < window_size:
        pad_len = window_size - sequence.shape[0]
        sequence = np.pad(sequence, ((0, pad_len), (0, 0)), mode="constant")

    windows: List[np.ndarray] = []
    for start in range(0, sequence.shape[0] - window_size + 1, stride):
        windows.append(sequence[start:start + window_size].astype(np.float32))

    if not windows:
        fallback = np.zeros((window_size, sequence.shape[1]), dtype=np.float32)
        valid_len = min(window_size, sequence.shape[0])
        fallback[:valid_len] = sequence[:valid_len]
        windows.append(fallback)

    return np.asarray(windows, dtype=np.float32)


def fetch_etherscan_transaction_record(address: str, limit: int = 100) -> WalletTransactionRecord:
    api_key = "BRI6G5PQGWG5VX1R8CIKP1MW1HJIXZE8GC"
    if not api_key:
        raise RuntimeError("Etherscan API key is missing. Set ETHERSCAN_API_KEY in the backend environment.")

    normalized_address = _normalize_address(address)
    response = requests.get(
        ETHERSCAN_API_URL,
        params={
            "chainid": 1,
            "module": "account",
            "action": "txlist",
            "address": normalized_address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": limit,
            "sort": "desc",
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    message = str(payload.get("message", ""))
    status = str(payload.get("status", ""))
    records = payload.get("result", [])

    if status != "1":
        if message == "No transactions found":
            records = []
        elif not isinstance(records, list):
            raise RuntimeError(str(records) or "Unable to fetch transaction history from Etherscan.")

    if not isinstance(records, list):
        raise RuntimeError("Unexpected Etherscan response format.")

    transactions = pd.DataFrame(records)
    if transactions.empty:
        summary: Dict[str, object] = {
            "source": "Etherscan",
            "total_transactions": 0,
            "sent_transactions": 0,
            "received_transactions": 0,
            "total_value_eth_sent": 0.0,
            "total_value_eth_received": 0.0,
            "first_transaction": None,
            "last_transaction": None,
            "unique_counterparties": 0,
        }
        return WalletTransactionRecord(address=normalized_address, summary=summary, transactions=transactions)

    transactions = transactions.copy()
    transactions["from"] = transactions["from"].astype(str).map(_normalize_address)
    transactions["to"] = transactions["to"].astype(str).map(_normalize_address)
    transactions["timestamp_unix"] = transactions["timeStamp"].map(_to_float)
    transactions["timeStamp"] = pd.to_datetime(transactions["timestamp_unix"], unit="s", errors="coerce")
    transactions["value_wei"] = transactions["value"].map(_to_float)
    transactions["value_eth"] = transactions["value_wei"] / 1e18
    transactions["block_number"] = transactions["blockNumber"].map(_to_float)

    sent_mask = transactions["from"] == normalized_address
    recv_mask = transactions["to"] == normalized_address
    counterparties = pd.concat([transactions.loc[sent_mask, "to"], transactions.loc[recv_mask, "from"]], ignore_index=True)

    summary = {
        "source": "Etherscan",
        "total_transactions": int(len(transactions)),
        "sent_transactions": int(sent_mask.sum()),
        "received_transactions": int(recv_mask.sum()),
        "total_value_eth_sent": float(transactions.loc[sent_mask, "value_eth"].sum()),
        "total_value_eth_received": float(transactions.loc[recv_mask, "value_eth"].sum()),
        "first_transaction": transactions["timeStamp"].min(),
        "last_transaction": transactions["timeStamp"].max(),
        "unique_counterparties": int(counterparties.nunique()),
    }

    display_columns = [
        col
        for col in ["timeStamp", "hash", "from", "to", "value_wei", "value_eth", "block_number", "gas", "gasPrice", "isError"]
        if col in transactions.columns
    ]
    return WalletTransactionRecord(
        address=normalized_address,
        summary=summary,
        transactions=transactions[display_columns + ["timestamp_unix"]],
    )


class EthereumPhishingPredictor:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.etherscan_api_key = "BRI6G5PQGWG5VX1R8CIKP1MW1HJIXZE8GC"

        self.basic_dim: Optional[int] = None
        self.gae_in_dim: Optional[int] = None
        self.edge_dim: int = 3

        self.dahgnn_model: Optional[DAHGNN] = None
        self.gae_model: Optional[GAE_PDNA] = None
        self.rf_model: Optional[RandomForestClassifier] = None

        self._load_dahgnn()
        self._load_gae_rf()

    def _resolve(self, filename: str) -> Path:
        return self.base_dir / filename

    def fetch_etherscan_transactions(self, address: str, limit: int = 100) -> WalletTransactionRecord:
        return fetch_etherscan_transaction_record(address, limit=limit)

    def _infer_basic_dim_from_dahgnn(self, state_dict: Dict[str, torch.Tensor]) -> int:
        key = "temporal.conv_branch.conv.weight"
        if key in state_dict and state_dict[key].ndim == 3:
            return int(state_dict[key].shape[1])

        cls_key = "classifier.0.weight"
        if cls_key in state_dict and state_dict[cls_key].ndim == 2:
            combined_dim = int(state_dict[cls_key].shape[1])
            return max(1, combined_dim - 96)

        raise RuntimeError("Unable to infer DA-HGNN input feature dimension from checkpoint.")

    def _infer_gae_in_channels(self, state_dict: Dict[str, torch.Tensor], fallback: int) -> int:
        probe_keys = [
            "encoder.blocks.0.conv.lin.weight",
            "encoder.blocks.0.conv.weight",
        ]
        for key in probe_keys:
            if key in state_dict and state_dict[key].ndim == 2:
                return int(state_dict[key].shape[1])
        return fallback

    def _load_dahgnn(self) -> None:
        weights_file = self._resolve(DAHGNN_WEIGHTS_FILE)
        if not weights_file.exists():
            raise FileNotFoundError(f"Missing {DAHGNN_WEIGHTS_FILE}. Put your trained DA-HGNN checkpoint beside the app.")

        state_dict = torch.load(weights_file, map_location=self.device)
        self.basic_dim = self._infer_basic_dim_from_dahgnn(state_dict)

        self.dahgnn_model = DAHGNN(
            num_basic_features=self.basic_dim,
            temporal_seq_features=self.basic_dim,
            temporal_window=4,
            graph_hidden_dim=64,
            graph_out_dim=32,
        ).to(self.device)
        self.dahgnn_model.load_state_dict(state_dict)
        self.dahgnn_model.eval()

    def _load_gae_rf(self) -> None:
        if self.basic_dim is None:
            raise RuntimeError("DA-HGNN must be loaded before GAE model setup.")

        gae_weights_file = self._resolve(GAE_WEIGHTS_FILE)
        if not gae_weights_file.exists():
            raise FileNotFoundError(f"Missing {GAE_WEIGHTS_FILE}. Put your trained GAE checkpoint beside the app.")

        state_dict = torch.load(gae_weights_file, map_location=self.device)
        gae_in_channels = self._infer_gae_in_channels(state_dict, self.basic_dim)
        self.gae_in_dim = gae_in_channels

        self.gae_model = GAE_PDNA(
            in_channels=gae_in_channels,
            hidden_channels=32,
            out_channels=15,
            edge_dim=self.edge_dim,
            edge_hidden=6,
            num_blocks=4,
        ).to(self.device)
        self.gae_model.load_state_dict(state_dict)
        self.gae_model.eval()

        rf_file = self._resolve(RF_MODEL_FILE)
        if not rf_file.exists():
            raise FileNotFoundError(
                f"Missing {RF_MODEL_FILE}. Put your pre-trained RF classifier beside the app."
            )
        self.rf_model = joblib.load(rf_file)

    def _encode_stats(self, stats: Dict[str, float], out_dim: int) -> np.ndarray:
        total_tx = max(stats.get("total_tx", 0.0), 0.0)
        sent_tx = max(stats.get("sent_tx", 0.0), 0.0)
        recv_tx = max(stats.get("recv_tx", 0.0), 0.0)
        total_eth = max(stats.get("total_eth", 0.0), 0.0)
        sent_eth = max(stats.get("sent_eth", 0.0), 0.0)
        recv_eth = max(stats.get("recv_eth", 0.0), 0.0)
        unique_c = max(stats.get("unique_counterparties", 0.0), 0.0)
        avg_interval = max(stats.get("avg_interval", 0.0), 0.0)
        active_window = max(stats.get("active_window", 0.0), 0.0)
        burst = min(max(stats.get("burst_score", 0.0), 0.0), 1.0)

        denom = total_tx + 1e-6
        base = np.array([
            min(np.log1p(total_tx) / 8.0, 1.0),
            min(sent_tx / denom, 1.0),
            min(recv_tx / denom, 1.0),
            min(np.log1p(total_eth) / 10.0, 1.0),
            min(np.log1p(sent_eth) / 10.0, 1.0),
            min(np.log1p(recv_eth) / 10.0, 1.0),
            min(unique_c / 50.0, 1.0),
            min(np.log1p(avg_interval) / 10.0, 1.0),
            min(np.log1p(active_window) / 12.0, 1.0),
            burst,
        ], dtype=np.float32)

        out = np.zeros(out_dim, dtype=np.float32)
        copy_len = min(out_dim, base.shape[0])
        out[:copy_len] = base[:copy_len]
        return out

    def _compute_node_stats(self, node_address: str, tx_df: pd.DataFrame) -> Dict[str, float]:
        node = _normalize_address(node_address)
        involved = tx_df[(tx_df["from"] == node) | (tx_df["to"] == node)]
        if involved.empty:
            return {
                "total_tx": 0.0,
                "sent_tx": 0.0,
                "recv_tx": 0.0,
                "total_eth": 0.0,
                "sent_eth": 0.0,
                "recv_eth": 0.0,
                "unique_counterparties": 0.0,
                "avg_interval": 0.0,
                "active_window": 0.0,
                "burst_score": 0.0,
            }

        sent = involved[involved["from"] == node]
        recv = involved[involved["to"] == node]

        timestamps = involved["timestamp_unix"].dropna().astype(float).sort_values().values
        avg_interval = 0.0
        active_window = 0.0
        burst_score = 0.0
        if timestamps.size > 1:
            intervals = np.diff(timestamps)
            avg_interval = float(np.mean(intervals))
            active_window = float(max(timestamps[-1] - timestamps[0], 0.0))
            if active_window > 0:
                cutoff = timestamps[0] + (0.2 * active_window)
                burst_score = float(np.sum(timestamps <= cutoff) / len(timestamps))

        counterparties = pd.concat([sent["to"], recv["from"]], ignore_index=True).dropna()

        return {
            "total_tx": float(len(involved)),
            "sent_tx": float(len(sent)),
            "recv_tx": float(len(recv)),
            "total_eth": float(involved["value_eth"].sum()),
            "sent_eth": float(sent["value_eth"].sum()),
            "recv_eth": float(recv["value_eth"].sum()),
            "unique_counterparties": float(counterparties.nunique()),
            "avg_interval": avg_interval,
            "active_window": active_window,
            "burst_score": burst_score,
        }

    def _build_inference_context(
        self,
        address: str,
        tx_record: WalletTransactionRecord,
        max_counterparties: int = 40,
    ) -> InferenceContext:
        if self.basic_dim is None:
            raise RuntimeError("Model input feature dimension is unknown.")
        if self.gae_in_dim is None:
            raise RuntimeError("GAE input feature dimension is unknown.")

        wallet = _normalize_address(address)
        tx_df = tx_record.transactions.copy()
        if tx_df.empty:
            dahgnn_feature_vector = self._encode_stats({}, self.basic_dim)
            dahgnn_x = torch.tensor(dahgnn_feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            gae_feature_vector = self._encode_stats({}, self.gae_in_dim)
            gae_x = torch.tensor(gae_feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((1, self.edge_dim), dtype=torch.float32, device=self.device)
            return InferenceContext(
                dahgnn_feature_vector=dahgnn_feature_vector,
                dahgnn_node_features=dahgnn_x,
                gae_node_features=gae_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_idx=0,
            )

        for col in ["from", "to"]:
            if col in tx_df.columns:
                tx_df[col] = tx_df[col].astype(str).map(_normalize_address)

        wallet_rows = tx_df[(tx_df["from"] == wallet) | (tx_df["to"] == wallet)]
        cp_series = pd.concat([
            wallet_rows.loc[wallet_rows["from"] == wallet, "to"],
            wallet_rows.loc[wallet_rows["to"] == wallet, "from"],
        ], ignore_index=True)
        cp_counts = cp_series.value_counts()
        counterparties = [addr for addr in cp_counts.index.tolist() if addr and addr != wallet][:max_counterparties]

        node_addresses = [wallet] + counterparties
        node_index = {addr: i for i, addr in enumerate(node_addresses)}

        dahgnn_node_features_list: List[np.ndarray] = []
        gae_node_features_list: List[np.ndarray] = []
        for node_addr in node_addresses:
            stats = self._compute_node_stats(node_addr, tx_df)
            dahgnn_node_features_list.append(self._encode_stats(stats, self.basic_dim))
            gae_node_features_list.append(self._encode_stats(stats, self.gae_in_dim))

        dahgnn_x_np = np.stack(dahgnn_node_features_list, axis=0).astype(np.float32)
        dahgnn_x = torch.tensor(dahgnn_x_np, dtype=torch.float32, device=self.device)
        gae_x_np = np.stack(gae_node_features_list, axis=0).astype(np.float32)
        gae_x = torch.tensor(gae_x_np, dtype=torch.float32, device=self.device)

        src: List[int] = []
        dst: List[int] = []
        edge_attrs: List[List[float]] = []

        for _, row in tx_df.iterrows():
            frm = _normalize_address(row.get("from", ""))
            to = _normalize_address(row.get("to", ""))
            if frm in node_index and to in node_index:
                src.append(node_index[frm])
                dst.append(node_index[to])
                edge_attrs.append([
                    _to_float(row.get("timestamp_unix", 0.0)),
                    _to_float(row.get("value_wei", row.get("value", 0.0))),
                    _to_float(row.get("block_number", row.get("blockNumber", 0.0))),
                ])

        if not src:
            src = [0]
            dst = [0]
            edge_attrs = [[0.0, 0.0, 0.0]]

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_attr_np = _minmax_scale(np.asarray(edge_attrs, dtype=np.float32))
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=self.device)

        return InferenceContext(
            dahgnn_feature_vector=dahgnn_x_np[0],
            dahgnn_node_features=dahgnn_x,
            gae_node_features=gae_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_idx=0,
        )

    def predict_dahgnn(
        self,
        address: str,
        context: InferenceContext,
    ) -> Tuple[float, str]:
        if self.dahgnn_model is None:
            raise RuntimeError("DA-HGNN model is not loaded.")

        feature_vector = context.dahgnn_feature_vector
        node_features = context.dahgnn_node_features
        edge_index = context.edge_index
        node_idx = context.node_idx
        windows = make_node_windows(feature_vector, window_size=4, stride=2)

        basic_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_idx_tensor = torch.tensor([node_idx], dtype=torch.long, device=self.device)

        window_probs: List[float] = []
        with torch.no_grad():
            for window in windows:
                time_tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
                prob = self.dahgnn_model(
                    time_tensor,
                    node_features,
                    edge_index,
                    basic_tensor,
                    node_idx_tensor,
                )
                window_probs.append(float(prob.squeeze().detach().cpu().item()))

        probability = float(np.mean(window_probs)) if window_probs else 0.0
        label = "Phishing" if probability >= 0.5 else "Benign"
        return probability, label

    def predict_gae_rf(
        self,
        address: str,
        context: InferenceContext,
    ) -> Tuple[float, str]:
        if self.gae_model is None or self.rf_model is None:
            raise RuntimeError("GAE or RF model is not loaded.")

        node_features = context.gae_node_features
        edge_index = context.edge_index
        edge_attr = context.edge_attr
        node_idx = context.node_idx
        with torch.no_grad():
            embeddings = self.gae_model.encode(node_features, edge_index, edge_attr)
            node_embedding = embeddings[node_idx].detach().cpu().numpy().reshape(1, -1)

        probability = float(self.rf_model.predict_proba(node_embedding)[0, 1])
        label = "Phishing" if probability >= 0.5 else "Benign"
        return probability, label

    def predict(self, address: str, tx_record: Optional[WalletTransactionRecord] = None) -> PredictionResult:
        tx = tx_record or self.fetch_etherscan_transactions(address)
        context = self._build_inference_context(address, tx)

        dahgnn_probability, dahgnn_label = self.predict_dahgnn(address, context)
        gae_rf_probability, gae_rf_label = self.predict_gae_rf(address, context)
        final_probability = float((dahgnn_probability * 0.4 + gae_rf_probability * 0.6))
        final_label = "Phishing" if final_probability >= 0.5 else "Benign"

        return PredictionResult(
            address=_normalize_address(address),
            dahgnn_probability=dahgnn_probability,
            dahgnn_label=dahgnn_label,
            gae_rf_probability=gae_rf_probability,
            gae_rf_label=gae_rf_label,
            final_probability=final_probability,
            final_label=final_label,
        )


def load_predictor(base_dir: Optional[str] = None) -> EthereumPhishingPredictor:
    resolved_base = Path(base_dir or os.getcwd())
    return EthereumPhishingPredictor(resolved_base)
