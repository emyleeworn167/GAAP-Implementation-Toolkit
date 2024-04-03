"""Transaction network analysis for cross-border fraud detection.

Builds a directed graph of cross-border transactions and detects structural
anomalies that indicate fraud patterns:

- Circular flows (round-tripping): A→B→C→A disguises income origin
- Hub entities: Nodes with disproportionate centrality may be conduit companies
- Layering: Complex multi-hop paths between related endpoints obscure the
  true economic substance of transactions
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class CircularFlow:
    """A detected circular transaction flow."""

    entities: list[str]
    total_amount: float
    avg_amount: float
    amount_variance: float
    length: int


@dataclass
class TransactionNetwork:
    """Analyze cross-border transaction patterns using graph methods.

    Parameters
    ----------
    amount_col : str
        Column name for transaction amounts.
    sender_col : str
        Column name for sending entity.
    receiver_col : str
        Column name for receiving entity.
    """

    amount_col: str = "amount"
    sender_col: str = "sender"
    receiver_col: str = "receiver"

    def build_graph(self, transactions: pd.DataFrame) -> nx.DiGraph:
        """Build a weighted directed graph from transaction records."""
        g = nx.DiGraph()

        for _, row in transactions.iterrows():
            sender = row[self.sender_col]
            receiver = row[self.receiver_col]
            amount = row[self.amount_col]

            if g.has_edge(sender, receiver):
                g[sender][receiver]["weight"] += amount
                g[sender][receiver]["count"] += 1
                g[sender][receiver]["amounts"].append(amount)
            else:
                g.add_edge(
                    sender, receiver, weight=amount, count=1, amounts=[amount]
                )

        return g

    def detect_circular_flows(
        self,
        transactions: pd.DataFrame,
        max_cycle_length: int = 5,
        min_amount: float = 0.0,
    ) -> list[CircularFlow]:
        """Detect circular transaction flows (round-tripping indicators).

        Uses cycle detection on the transaction graph. Cycles with similar
        amounts across edges are particularly suspicious as they suggest
        coordinated fund movement.
        """
        g = self.build_graph(transactions)
        cycles: list[CircularFlow] = []
        seen_cycles: set[tuple] = set()

        for cycle in nx.simple_cycles(g, length_bound=max_cycle_length):
            # Normalize cycle representation to avoid duplicates
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            if normalized in seen_cycles:
                continue
            seen_cycles.add(normalized)

            # Compute cycle edge amounts
            amounts = []
            for i in range(len(cycle)):
                src = cycle[i]
                dst = cycle[(i + 1) % len(cycle)]
                if g.has_edge(src, dst):
                    amounts.append(g[src][dst]["weight"])

            total = sum(amounts)
            if total < min_amount:
                continue

            avg = np.mean(amounts)
            var = float(np.var(amounts) / max(avg**2, 1e-10))  # coefficient of variation squared

            cycles.append(
                CircularFlow(
                    entities=list(cycle),
                    total_amount=round(total, 2),
                    avg_amount=round(avg, 2),
                    amount_variance=round(var, 6),
                    length=len(cycle),
                )
            )

        # Sort by suspiciousness: low variance = more coordinated = more suspicious
        cycles.sort(key=lambda c: c.amount_variance)
        return cycles

    def hub_analysis(self, transactions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Identify hub entities with disproportionate transaction centrality.

        Hub entities in cross-border networks may indicate conduit companies
        used to layer transactions and obscure economic substance.
        """
        g = self.build_graph(transactions)

        rows = []
        in_degree = dict(g.in_degree(weight="weight"))
        out_degree = dict(g.out_degree(weight="weight"))
        betweenness = nx.betweenness_centrality(g, weight="weight")

        for node in g.nodes():
            rows.append(
                {
                    "entity": node,
                    "in_flow": round(in_degree.get(node, 0), 2),
                    "out_flow": round(out_degree.get(node, 0), 2),
                    "net_flow": round(
                        in_degree.get(node, 0) - out_degree.get(node, 0), 2
                    ),
                    "betweenness_centrality": round(betweenness.get(node, 0), 6),
                    "in_partners": g.in_degree(node),
                    "out_partners": g.out_degree(node),
                }
            )

        result = pd.DataFrame(rows)
        result = result.sort_values("betweenness_centrality", ascending=False)
        return result.head(top_n).reset_index(drop=True)
