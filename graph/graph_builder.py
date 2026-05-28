import networkx as nx
from typing import List, Tuple

from torch_geometric.data import HeteroData
import torch
import numpy as np

from simulation.topology.nodes import (
    Node,
    NodeType,
    generate_nodes,
)

from simulation.network.models import PathLossModel


def build_graph(
    num_sats=3,
    num_uavs=5,
    num_ground=10,
    num_clients=20,
    area_size=2000,
) -> Tuple[nx.Graph, List[Node]]:

    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
    )

    graph = nx.Graph()

    for node in nodes:

        graph.add_node(
            node.id,
            node_type=node.type.value,
            position=node.position.tolist(),
        )

    network_model = PathLossModel()

    for i, source in enumerate(nodes):

        for target in nodes[i + 1:]:

            try:
                latency = network_model.latency(
                    source,
                    target,
                )

            except Exception:
                continue

            graph.add_edge(
                source.id,
                target.id,
                latency=float(latency),
            )

    return graph, nodes


def build_graph_snapshot(
    candidates,
    clients,
    selected_servers=None,
):

    if selected_servers is None:
        selected_servers = []

    data = HeteroData()

    type_map = {
        NodeType.SATELLITE: "satellite",
        NodeType.UAV: "hap",
        NodeType.GROUND: "ground",
        NodeType.CLIENT: "client",
    }

    node_to_local = {}

    all_nodes = list(candidates) + list(clients)

    for t in type_map.values():
        data[t].x = []

    for n in all_nodes:

        tname = type_map[n.type]

        type_feat = {
            NodeType.SATELLITE: [1, 0, 0, 0],
            NodeType.UAV: [0, 1, 0, 0],
            NodeType.GROUND: [0, 0, 1, 0],
            NodeType.CLIENT: [0, 0, 0, 1],
        }[n.type]

        pos = np.asarray(n.position)

        feat = np.array(
            [
                *pos,
                np.log10(max(n.power, 1e-9)),
                np.log10(max(n.noise_variance, 1e-12)),
                *type_feat,
                float(n in selected_servers),
            ],
            dtype=np.float32,
        )

        node_to_local[n.id] = (
            tname,
            len(data[tname].x),
        )

        data[tname].x.append(feat)

    for t in type_map.values():

        if len(data[t].x) == 0:
            continue

        data[t].x = torch.tensor(
            np.array(data[t].x),
            dtype=torch.float32,
        )

    edge_store = {}
    edge_attr_store = {}

    for c in clients:

        for s in candidates:

            try:
                snr = c.compute_snr_to(s)

            except Exception:
                continue

            if not np.isfinite(snr):
                continue

            src_t, src_i = node_to_local[c.id]
            dst_t, dst_i = node_to_local[s.id]

            et = (src_t, "link", dst_t)

            edge_store.setdefault(et, [[], []])

            edge_store[et][0].append(src_i)
            edge_store[et][1].append(dst_i)

            edge_attr_store.setdefault(et, [])

            edge_attr_store[et].append([
                float(np.log10(snr + 1e-9))
            ])

    for et, eidx in edge_store.items():

        data[et].edge_index = torch.tensor(
            eidx,
            dtype=torch.long,
        )

        data[et].edge_attr = torch.tensor(
            edge_attr_store[et],
            dtype=torch.float32,
        )

        rev_et = (et[2], "rev_link", et[0])

        rev_edge_index = torch.stack(
            [
                data[et].edge_index[1],
                data[et].edge_index[0],
            ],
            dim=0,
        )

        data[rev_et].edge_index = rev_edge_index

        data[rev_et].edge_attr = (
            data[et].edge_attr.clone()
        )

    node_maps = {}

    for t in ["satellite", "hap", "ground"]:

        node_maps[t] = {}

        idx = 0

        for n in candidates:

            if type_map[n.type] == t:
                node_maps[t][idx] = n
                idx += 1

    return data, node_maps