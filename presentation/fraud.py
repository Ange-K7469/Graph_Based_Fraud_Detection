"""
Fraud Intelligence & Knowledge Graph Explorer
==============================================

Streamlit dashboard for the "Dynamic Fraud Intelligence System" project
(see Fraud_Knowledge_Graph_Final_Project_Proposal.pdf).

Expected data (produced by the export cells appended to the end of
UnsupervisedKG.ipynb):
  - knowledge_graph.pkl                 -> NetworkX graph (pickle)
  - scored_validation_transactions.csv  -> scored transactions table

If these files are not present in the current folder, the app builds a
small synthetic knowledge graph on the fly (with a couple of made-up fraud
rings) so the interface can still be used/demoed before the real pipeline
has been run end to end.

Pages:
  - Overview                : overall graph metrics
  - Transaction Lookup       : lookup + explainability panel (Explainability Panel, PDF sec. 13)
  - Graph Explorer           : interactive subgraph around a node (Graph Explorer, sec. 13)
  - Community Explorer       : most suspicious communities + their subgraph (sec. 7, 13)
  - Report Suspicious Activity: report form with matching, new-node creation,
                                 dynamic risk recalculation and propagation (sec. 8-11)
  - Live Updates              : NEW NODE / RISK UPDATED / CLASS CHANGED log (sec. 13, "Live Updates")
"""

import copy
import os
import random
from collections import Counter, defaultdict, deque
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

try:
    import community as community_louvain
except ImportError:  # pragma: no cover
    community_louvain = None

import kg_risk as kr

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Intelligence & KG Explorer",
    page_icon="🛡️",
    layout="wide",
)

GRAPH_PATH = "knowledge_graph.pkl"
CSV_PATH = "scored_validation_transactions.csv"
MAX_SUBGRAPH_NODES = 120

NODE_TYPE_LABELS = {
    "Transaction": "Transaction",
    "FraudReport": "Report",
    "ExternalEntity": "External entity",
    "card1": "Card (card1)", "card2": "Card (card2)", "card3": "Card (card3)",
    "card4": "Card network", "card5": "Card (card5)", "card6": "Card type",
    "DeviceInfo": "Device", "DeviceType": "Device type",
    "id_31": "Browser", "id_30": "Operating system",
    "P_emaildomain": "Email domain",
}

ENTITY_COLOR = "#72B7B2"
EXTERNAL_COLOR = "#9D6BB0"
REPORT_COLOR = "#B279A2"
RISK_COLORS = {
    kr.RISK_LOW: "#54A24B",
    kr.RISK_SUSPICIOUS: "#F2B701",
    kr.RISK_HIGH: "#E45756",
}


# ---------------------------------------------------------------------------
# Demo data generation (fallback for when the real artifacts are missing)
# ---------------------------------------------------------------------------
def _build_demo_graph(n_normal=140, seed=7):
    rng = random.Random(seed)
    G = nx.Graph()
    tx_counter = [3_000_000]

    def add_tx(amt, is_fraud):
        tx_counter[0] += 1
        tx_id = tx_counter[0]
        G.add_node(
            tx_id, node_type="Transaction", amount=amt, is_fraud=is_fraud,
            timestamp=tx_id, is_validation=True,
        )
        return tx_id

    def link(tx, col, val):
        node_id = kr.entity_node_id(col, val)
        if not G.has_node(node_id):
            G.add_node(node_id, node_type=col, frequency=0)
        G.nodes[node_id]["frequency"] += 1
        G.add_edge(tx, node_id, relationship=col)

    devices = [f"dev-{i}" for i in range(40)]
    emails = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com", "corp-mail.biz"]
    cards = list(range(10000, 10070))
    browsers = ["chrome 96", "safari 15", "firefox 94", "mobile safari"]
    systems = ["Windows 10", "iOS 15", "Android 11", "Mac OS X"]
    card_types = ["visa", "mastercard", "amex", "discover"]

    normal_tx = []
    for _ in range(n_normal):
        amt = round(rng.uniform(10, 500), 2)
        is_fraud = 1 if rng.random() < 0.02 else 0
        tx = add_tx(amt, is_fraud)
        link(tx, "card1", rng.choice(cards))
        link(tx, "DeviceInfo", rng.choice(devices))
        link(tx, "P_emaildomain", rng.choice(emails))
        link(tx, "id_31", rng.choice(browsers))
        link(tx, "id_30", rng.choice(systems))
        link(tx, "card4", rng.choice(card_types))
        normal_tx.append(tx)

    # A few "fraud rings" that deliberately share rare devices/emails
    # (to illustrate the PDF's rule: rare shared entities = strong evidence)
    ring_tx_ids = []
    for ring_i in range(3):
        ring_device = f"ring-device-{ring_i}"
        ring_email = f"ring{ring_i}@tempmail.top"
        ring_cards = [90000 + ring_i * 10 + k for k in range(3)]
        for _ in range(rng.randint(5, 9)):
            amt = round(rng.uniform(300, 2000), 2)
            tx = add_tx(amt, 1)
            link(tx, "DeviceInfo", ring_device)
            link(tx, "P_emaildomain", ring_email)
            link(tx, "card1", rng.choice(ring_cards))
            link(tx, "id_31", "headless-chrome")
            ring_tx_ids.append(tx)

    ring_set = set(ring_tx_ids)
    n_tx_total = n_normal + len(ring_tx_ids)
    hub_threshold = max(3, int(n_tx_total * 0.05))
    for u, v, edata in G.edges(data=True):
        entity = u if G.nodes[u]["node_type"] != "Transaction" else v
        freq = G.nodes[entity].get("frequency", 1)
        edata["weight"] = 0.05 / freq if freq > hub_threshold else 1.0 / (1.0 + np.log(freq))

    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
    if community_louvain is not None:
        partition = community_louvain.best_partition(G, weight="weight")
    else:  # pragma: no cover - fallback if python-louvain isn't installed
        partition = {n: i for i, comp in enumerate(nx.connected_components(G)) for n in comp}
    comm_counts = Counter(partition.values())

    raw_scores = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Transaction":
            continue
        neighbors = list(G.neighbors(node))
        weighted_deg = sum(G[node][n].get("weight", 1.0) for n in neighbors)
        base = weighted_deg + (data["amount"] / 2000.0) * 0.3
        if node in ring_set:
            base += rng.uniform(0.8, 1.6)
        base += rng.uniform(-0.15, 0.15)
        raw_scores[node] = max(0.0, base)
        neighbor_comms = [partition.get(n) for n in neighbors if n in partition]
        data["cluster_id"] = (
            max(set(neighbor_comms), key=neighbor_comms.count) if neighbor_comms else partition.get(node, -1)
        )
        data["graph_degree"] = G.degree(node)
        data["weighted_degree"] = weighted_deg
        data["pagerank"] = pagerank.get(node, 0.0)
        data["community_size"] = comm_counts.get(data["cluster_id"], 1)

    lo, hi = min(raw_scores.values()), max(raw_scores.values())
    for node, raw in raw_scores.items():
        G.nodes[node]["anomaly_score"] = raw
        G.nodes[node]["anomaly_evidence"] = (raw - lo) / (hi - lo) if hi > lo else 0.0

    cluster_vals = defaultdict(list)
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "Transaction":
            cluster_vals[data["cluster_id"]].append(data["anomaly_evidence"])
    cluster_mean = {c: sum(v) / len(v) for c, v in cluster_vals.items()}

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Transaction":
            continue
        neighbour_risk = cluster_mean.get(data["cluster_id"], data["anomaly_evidence"])
        data["risk_score"] = kr.transaction_baseline_risk(data["anomaly_evidence"], neighbour_risk)
        data["risk_class"] = kr.risk_class(data["risk_score"])
        data["report_evidence"] = 0.0
        data["num_reports"] = 0

    for node, data in G.nodes(data=True):
        if data.get("node_type") not in kr.ENTITY_COLS:
            continue
        nbh_scores = [
            G.nodes[n]["risk_score"] for n in G.neighbors(node)
            if G.nodes[n].get("node_type") == "Transaction"
        ]
        neighbour_risk = sum(nbh_scores) / len(nbh_scores) if nbh_scores else 0.0
        data["risk_score"] = kr.entity_risk(0.0, neighbour_risk)
        data["risk_class"] = kr.risk_class(data["risk_score"])
        data["report_evidence"] = 0.0
        data["num_reports"] = 0
        data["community_id"] = partition.get(node, -1)

    return G


@st.cache_resource(show_spinner="Preparing the knowledge graph...")
def _load_pristine_graph_and_source():
    if os.path.exists(GRAPH_PATH):
        import pickle
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
        return G, "real"
    return _build_demo_graph(), "demo"


def build_tx_dataframe(G):
    rows = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "Transaction":
            continue
        rows.append({
            "TransactionID": node,
            "TransactionAmt": data.get("amount", 0.0),
            "isFraud": data.get("is_fraud", 0),
            "anomaly_evidence": data.get("anomaly_evidence", 0.0),
            "cluster_id": data.get("cluster_id", -1),
            "risk_score": data.get("risk_score", 0.0),
            "risk_class": data.get("risk_class", kr.RISK_LOW),
            "report_evidence": data.get("report_evidence", 0.0),
            "num_reports": data.get("num_reports", 0),
        })
    return pd.DataFrame(rows)


def initialize_session():
    if "G" in st.session_state:
        return
    pristine_G, source = _load_pristine_graph_and_source()
    st.session_state.G = copy.deepcopy(pristine_G)
    st.session_state.data_source = source
    st.session_state.events = []
    st.session_state.report_counter = 0
    if source == "real" and os.path.exists(CSV_PATH):
        st.session_state.tx_df = pd.read_csv(CSV_PATH)
    else:
        st.session_state.tx_df = build_tx_dataframe(st.session_state.G)


def reset_session():
    for key in ["G", "data_source", "events", "report_counter", "tx_df"]:
        st.session_state.pop(key, None)
    initialize_session()


# ---------------------------------------------------------------------------
# Node resolution / dynamic report-handling logic (PDF sections 8-11)
# ---------------------------------------------------------------------------
def resolve_target_node(G, entity_choice, raw_value):
    """Returns (node_id, is_new, error)."""
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return None, False, "Please enter a value."

    col = kr.REPORTABLE_ENTITIES[entity_choice]

    if col == "__transaction__":
        try:
            node_id = int(raw_value)
        except ValueError:
            return None, False, "TransactionID must be a number."
        if node_id not in G:
            return None, False, "TransactionID not found in the graph."
        return node_id, False, None

    if col == "__external__":
        node_id = kr.external_node_id(raw_value)
        return node_id, node_id not in G, None

    value = raw_value
    if col in ("card1", "card2", "card3", "card5"):
        try:
            value = int(raw_value)
        except ValueError:
            return None, False, f"{col} must be a number."
    node_id = kr.entity_node_id(col, value)
    return node_id, node_id not in G, None


def resolve_lookup_node(G, entity_choice, raw_value):
    node_id, is_new, err = resolve_target_node(G, entity_choice, raw_value)
    if err:
        return None, err
    if is_new:
        return None, "Node not found in the graph."
    return node_id, None


def sync_tx_row(tx_id):
    G = st.session_state.G
    data = G.nodes[tx_id]
    row = {
        "TransactionID": tx_id,
        "TransactionAmt": data.get("amount", 0.0),
        "isFraud": data.get("is_fraud", 0),
        "anomaly_evidence": data.get("anomaly_evidence", 0.0),
        "cluster_id": data.get("cluster_id", -1),
        "risk_score": data.get("risk_score", 0.0),
        "risk_class": data.get("risk_class", kr.RISK_LOW),
        "report_evidence": data.get("report_evidence", 0.0),
        "num_reports": data.get("num_reports", 0),
    }
    df = st.session_state.tx_df
    match = df.index[df["TransactionID"] == tx_id]
    if len(match):
        for k, v in row.items():
            if k in df.columns:
                df.at[match[0], k] = v
    else:
        st.session_state.tx_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def _node_report_evidence(G, node_id):
    confidences = [
        G.nodes[r].get("confidence", 0.0)
        for r in G.neighbors(node_id)
        if G.nodes[r].get("node_type") == "FraudReport"
    ]
    return kr.noisy_or(confidences)


def _recompute_risk(G, node_id):
    data = G.nodes[node_id]
    report_evidence = _node_report_evidence(G, node_id)
    num_reports = sum(
        1 for r in G.neighbors(node_id) if G.nodes[r].get("node_type") == "FraudReport"
    )
    neighbour_scores = [
        G.nodes[n]["risk_score"] for n in G.neighbors(node_id)
        if G.nodes[n].get("node_type") != "FraudReport" and "risk_score" in G.nodes[n]
    ]
    neighbour_risk = (
        sum(neighbour_scores) / len(neighbour_scores) if neighbour_scores else data.get("risk_score", 0.0)
    )

    if data.get("node_type") == "Transaction":
        new_score = kr.transaction_risk(data.get("anomaly_evidence", 0.0), report_evidence, neighbour_risk)
    else:
        new_score = kr.entity_risk(report_evidence, neighbour_risk)

    data["report_evidence"] = report_evidence
    data["num_reports"] = num_reports
    return new_score


def _recompute_and_propagate(G, origin, events, now):
    old_score = G.nodes[origin].get("risk_score", 0.0)
    old_class = G.nodes[origin].get("risk_class", kr.RISK_LOW)
    new_score = _recompute_risk(G, origin)
    G.nodes[origin]["risk_score"] = new_score
    new_class = kr.risk_class(new_score)
    G.nodes[origin]["risk_class"] = new_class

    if abs(new_score - old_score) >= kr.PROPAGATION_MIN_DELTA:
        events.append({"type": "RISK_UPDATED", "node": origin, "before": old_score, "after": new_score, "time": now})
    if new_class != old_class:
        events.append({"type": "CLASS_CHANGED", "node": origin, "before": old_class, "after": new_class, "time": now})

    # Controlled propagation to neighbours, decaying with hop distance (PDF section 11)
    visited = {origin}
    frontier = deque([(origin, 0)])
    while frontier:
        node, hop = frontier.popleft()
        if hop >= kr.PROPAGATION_MAX_HOPS:
            continue
        for nb in G.neighbors(node):
            if nb in visited or G.nodes[nb].get("node_type") == "FraudReport":
                continue
            visited.add(nb)
            edge_weight = G[node][nb].get("weight", 0.3)
            influence = kr.propagation_influence(G.nodes[node]["risk_score"], edge_weight, hop + 1)
            if influence < kr.PROPAGATION_MIN_DELTA:
                continue
            nb_old_score = G.nodes[nb].get("risk_score", 0.0)
            nb_old_class = G.nodes[nb].get("risk_class", kr.RISK_LOW)
            nb_new_score = kr.apply_influence(nb_old_score, influence)
            if abs(nb_new_score - nb_old_score) >= kr.PROPAGATION_MIN_DELTA:
                G.nodes[nb]["risk_score"] = nb_new_score
                nb_new_class = kr.risk_class(nb_new_score)
                G.nodes[nb]["risk_class"] = nb_new_class
                events.append({
                    "type": "RISK_UPDATED", "node": nb, "before": nb_old_score, "after": nb_new_score,
                    "time": now, "via_propagation": True,
                })
                if nb_new_class != nb_old_class:
                    events.append({
                        "type": "CLASS_CHANGED", "node": nb, "before": nb_old_class, "after": nb_new_class,
                        "time": now, "via_propagation": True,
                    })
            frontier.append((nb, hop + 1))


def apply_report(entity_choice, raw_value, reason_label, description):
    G = st.session_state.G
    node_id, is_new, err = resolve_target_node(G, entity_choice, raw_value)
    if err:
        return None, err

    events = []
    now = datetime.now().strftime("%H:%M:%S")
    col = kr.REPORTABLE_ENTITIES[entity_choice]
    node_type_for_new = "ExternalEntity" if col == "__external__" else col

    if is_new:
        G.add_node(
            node_id, node_type=node_type_for_new, risk_score=0.0, risk_class=kr.RISK_LOW,
            report_evidence=0.0, num_reports=0, frequency=0,
        )
        events.append({"type": "NEW_NODE", "node": node_id, "node_type": node_type_for_new, "time": now})

    st.session_state.report_counter += 1
    report_id = f"Report_{st.session_state.report_counter}"
    confidence = kr.REPORT_CONFIDENCE_WEIGHTS[reason_label]
    G.add_node(
        report_id, node_type="FraudReport", report_id=report_id, timestamp=now,
        report_type=reason_label, confidence=confidence, description=description,
    )
    G.add_edge(report_id, node_id, relationship="REPORTS", weight=confidence)
    events.append({
        "type": "REPORT_ADDED", "node": node_id, "report_id": report_id,
        "confidence": confidence, "time": now,
    })

    _recompute_and_propagate(G, node_id, events, now)

    st.session_state.events = events + st.session_state.events
    for ev in events:
        n = ev.get("node")
        if n is not None and G.has_node(n) and G.nodes[n].get("node_type") == "Transaction":
            sync_tx_row(n)

    return node_id, None


# ---------------------------------------------------------------------------
# Bounded subgraphs and PyVis rendering
# ---------------------------------------------------------------------------
def ego_subgraph_bounded(G, center, radius=1, max_nodes=MAX_SUBGRAPH_NODES):
    if center not in G:
        return nx.Graph(), False
    visited = {center}
    frontier = [center]
    truncated = False
    for _ in range(radius):
        candidates = []
        seen = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in visited and nb not in seen:
                    candidates.append(nb)
                    seen.add(nb)
        # Prefer rarer (less frequent) entities: per the PDF, they're the
        # most informative evidence, so we avoid filling the subgraph with
        # very common hubs.
        candidates.sort(key=lambda n: G.nodes[n].get("frequency", 1))
        next_frontier = []
        for nb in candidates:
            if len(visited) >= max_nodes:
                truncated = True
                break
            visited.add(nb)
            next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break
    return G.subgraph(visited).copy(), truncated


def node_color(data):
    node_type = data.get("node_type")
    if node_type == "FraudReport":
        return REPORT_COLOR
    if data.get("risk_class") in RISK_COLORS:
        return RISK_COLORS[data["risk_class"]]
    if node_type == "ExternalEntity":
        return EXTERNAL_COLOR
    return ENTITY_COLOR


def node_label(node_id, data):
    node_type = data.get("node_type")
    if node_type == "Transaction":
        return f"TX {node_id}"
    if node_type == "FraudReport":
        return f"📝 {node_id}"
    return str(node_id)


def render_graph_html(sub, center=None, height="560px"):
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=110, spring_strength=0.02, damping=0.9)
    for node, data in sub.nodes(data=True):
        color = node_color(data)
        is_center = node == center
        size = 30 if is_center else (18 if data.get("node_type") == "Transaction" else 14)
        lines = [f"type: {NODE_TYPE_LABELS.get(data.get('node_type'), data.get('node_type'))}"]
        if "risk_score" in data:
            lines.append(f"risk_score: {data['risk_score']:.2f} ({data.get('risk_class', '?')})")
        if data.get("node_type") == "Transaction":
            lines.append(f"amount: {data.get('amount', 0):.2f}")
            lines.append(f"isFraud (ground truth): {data.get('is_fraud', 0)}")
        if "num_reports" in data and data.get("num_reports"):
            lines.append(f"reports: {data.get('num_reports', 0)}")
        if data.get("node_type") == "FraudReport":
            lines.append(f"reason: {data.get('report_type', '')}")
            lines.append(f"confidence: {data.get('confidence', 0):.1f}")
        net.add_node(
            node, label=node_label(node, data), color=color, size=size,
            title="\n".join(lines), borderWidth=3 if is_center else 1,
            borderWidthSelected=4,
        )
    for u, v, edata in sub.edges(data=True):
        w = edata.get("weight", 0.3)
        net.add_edge(u, v, value=max(w, 0.05), title=edata.get("relationship", ""))
    return net.generate_html()


def show_graph(sub, center=None, truncated=False, height=560):
    if sub.number_of_nodes() == 0:
        st.info("No nodes to display.")
        return
    if truncated:
        st.caption(
            f"⚠️ Subgraph truncated to {sub.number_of_nodes()} nodes (showing the rarest/most informative "
            "entities) to keep it readable — this is not the full graph."
        )
    html = render_graph_html(sub, center=center, height=f"{height}px")
    components.html(html, height=height + 20, scrolling=True)


def risk_badge(risk_class):
    return f"{kr.RISK_CLASS_ICON.get(risk_class, '')} {risk_class}"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
initialize_session()
G = st.session_state.G
tx_df = st.session_state.tx_df

st.title("🛡️ Fraud Detection & Knowledge Graph Explorer")
st.markdown(
    "Look up transactions, explore the knowledge graph, and watch how a new report dynamically "
    "updates the risk and class of connected entities."
)

with st.sidebar:
    st.header("Navigation")
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Overview"
    page = st.radio(
        "Page",
        [
            "Overview",
            "Transaction Lookup",
            "Graph Explorer",
            "Community Explorer",
            "Report Suspicious Activity",
            "Live Updates",
        ],
        key="nav_page",
    )

    st.divider()
    source_label = "📡 real data (graph exported from the notebook)" if st.session_state.data_source == "real" else "🧪 synthetic demo data (no knowledge_graph.pkl found)"
    st.caption(f"Data source: {source_label}")
    st.caption(f"Nodes: {G.number_of_nodes():,} · Edges: {G.number_of_edges():,}")
    n_reports = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "FraudReport")
    st.caption(f"Reports on record: {n_reports} · Events logged: {len(st.session_state.events)}")
    if st.button("🔄 Reload data from scratch"):
        reset_session()
        st.rerun()

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.header("📊 Overview")

    n_entities = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") in kr.ENTITY_COLS + ["ExternalEntity"])
    class_counts = tx_df["risk_class"].value_counts().to_dict() if not tx_df.empty else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(tx_df):,}")
    c2.metric("Entities in graph", f"{n_entities:,}")
    c3.metric("🚨 High risk", f"{class_counts.get(kr.RISK_HIGH, 0):,}")
    c4.metric("⚠️ Suspicious", f"{class_counts.get(kr.RISK_SUSPICIOUS, 0):,}")

    st.subheader("Highest-risk transactions right now")
    if not tx_df.empty:
        top = tx_df.sort_values("risk_score", ascending=False).head(10)[
            ["TransactionID", "TransactionAmt", "risk_score", "risk_class", "num_reports", "isFraud"]
        ]
        top = top.rename(columns={
            "TransactionAmt": "Amount", "risk_score": "Risk score", "risk_class": "Class",
            "num_reports": "Reports", "isFraud": "isFraud (ground truth)",
        })
        st.dataframe(top, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions loaded.")

# ---------------------------------------------------------------------------
# Page: Transaction Lookup (Explainability Panel, PDF sec. 13)
# ---------------------------------------------------------------------------
elif page == "Transaction Lookup":
    st.header("🔍 Transaction Lookup & Explainability Panel")

    search_id = st.text_input("Enter a TransactionID:", placeholder="e.g. 3000012")

    if search_id:
        try:
            query_id = int(search_id)
        except ValueError:
            query_id = None

        if query_id is None or query_id not in G or G.nodes[query_id].get("node_type") != "Transaction":
            st.warning("No transaction found with this ID.")
        else:
            data = G.nodes[query_id]
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transaction ID", query_id)
                st.metric("Amount", f"${data.get('amount', 0):,.2f}")
            with col2:
                st.metric("Risk score", f"{data.get('risk_score', 0):.2%}")
                st.metric("Risk class", risk_badge(data.get("risk_class", kr.RISK_LOW)))
            with col3:
                status = "Fraudulent" if data.get("is_fraud", 0) == 1 else "Legitimate"
                st.metric("Historical ground truth", status)
                st.metric("Linked reports", data.get("num_reports", 0))

            st.subheader("🧠 Why this score?")
            parts = []
            anomaly = data.get("anomaly_evidence", 0.0)
            if anomaly > 0.6:
                parts.append(f"The model considers it **strongly anomalous** compared with typical behaviour (anomaly evidence: {anomaly:.0%}).")
            elif anomaly > 0.3:
                parts.append(f"The behaviour is moderately unusual (anomaly evidence: {anomaly:.0%}).")
            else:
                parts.append(f"The behaviour falls within normal patterns (anomaly evidence: {anomaly:.0%}).")
            if data.get("num_reports", 0) > 0:
                parts.append(
                    f"It is linked to **{data['num_reports']} community report{'s' if data['num_reports'] != 1 else ''}** "
                    f"(report evidence: {data.get('report_evidence', 0):.0%})."
                )
            parts.append(f"Belongs to graph community/cluster **{data.get('cluster_id', '?')}**.")
            text = " ".join(parts)
            if data.get("risk_score", 0) >= 0.7:
                st.error(text)
            elif data.get("risk_score", 0) >= 0.4:
                st.warning(text)
            else:
                st.success(text)

            st.subheader("Connected entities")
            neighbors = [n for n in G.neighbors(query_id)]
            if neighbors:
                rows = []
                for n in neighbors:
                    nd = G.nodes[n]
                    rows.append({
                        "Node": n,
                        "Type": NODE_TYPE_LABELS.get(nd.get("node_type"), nd.get("node_type")),
                        "Risk score": nd.get("risk_score"),
                        "Class": nd.get("risk_class"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No connected entities in the graph.")

            if st.button("🔗 View this transaction in the Graph Explorer"):
                st.session_state["ge_prefill_type"] = "Transaction (TransactionID)"
                st.session_state["ge_prefill_value"] = str(query_id)
                st.session_state.nav_page = "Graph Explorer"
                st.rerun()

# ---------------------------------------------------------------------------
# Page: Graph Explorer (PDF sec. 13)
# ---------------------------------------------------------------------------
elif page == "Graph Explorer":
    st.header("🕸️ Graph Explorer")
    st.caption(
        "Visualize the subgraph around a transaction or an entity: what's connected to what, "
        "and how risk changes as you move away from the central node."
    )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        entity_choice = st.selectbox(
            "Node type",
            list(kr.REPORTABLE_ENTITIES.keys()),
            index=0,
            key="ge_type",
        )
    with col_b:
        default_value = st.session_state.pop("ge_prefill_value", "")
        if st.session_state.pop("ge_prefill_type", None):
            pass  # the default type above stays as chosen; the user can still change it
        value = st.text_input("Value", value=default_value, placeholder="e.g. 3000012, a card number, gmail.com...")
    with col_c:
        radius = st.slider("Radius (hops)", 1, 2, 1)

    if value:
        node_id, err = resolve_lookup_node(G, entity_choice, value)
        if err:
            st.warning(err)
        else:
            data = G.nodes[node_id]
            info_cols = st.columns(4)
            info_cols[0].metric("Node", str(node_id))
            info_cols[1].metric("Type", NODE_TYPE_LABELS.get(data.get("node_type"), data.get("node_type")))
            info_cols[2].metric("Risk score", f"{data.get('risk_score', 0):.2f}")
            info_cols[3].metric("Class", risk_badge(data.get("risk_class", kr.RISK_LOW)))

            sub, truncated = ego_subgraph_bounded(G, node_id, radius=radius, max_nodes=MAX_SUBGRAPH_NODES)
            st.caption(f"Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges.")
            show_graph(sub, center=node_id, truncated=truncated)
    else:
        st.info("Enter a value to view its neighbourhood in the graph.")

    with st.expander("Color legend"):
        st.markdown(
            "- 🟢 Low risk · 🟡 Suspicious · 🔴 High risk (color = the node's risk class)\n"
            "- 🟣 Report (FraudReport)\n"
            "- Light purple = external entity created by a report (phone/URL/bank account/other)\n"
            "- The node with the thicker border is the search's central node"
        )

# ---------------------------------------------------------------------------
# Page: Community Explorer (PDF sec. 7 and 13)
# ---------------------------------------------------------------------------
elif page == "Community Explorer":
    st.header("👥 Community Explorer")
    st.caption(
        "Communities detected with Louvain on the weighted graph. Very large communities (likely a "
        "single very common hub, not a real fraud pattern) are excluded from the ranking to keep it "
        "readable — consistent with the team's simplification plan to show only targeted subgraphs."
    )

    if tx_df.empty:
        st.info("No transactions loaded.")
    else:
        max_meaningful_size = st.slider(
            "Maximum size for a community to be considered 'interesting'", 5, 2000, 500, step=5
        )
        agg = tx_df.groupby("cluster_id").agg(
            size=("TransactionID", "count"),
            avg_risk=("risk_score", "mean"),
            ground_truth_fraud_rate=("isFraud", "mean"),
            n_high_risk=("risk_class", lambda s: (s == kr.RISK_HIGH).sum()),
        ).reset_index()
        agg = agg[agg["size"] <= max_meaningful_size].sort_values("avg_risk", ascending=False)

        st.dataframe(agg.head(20), use_container_width=True, hide_index=True)

        if not agg.empty:
            options = agg["cluster_id"].head(20).tolist()
            selected_cluster = st.selectbox("Community to visualize", options)

            members = tx_df[tx_df["cluster_id"] == selected_cluster].sort_values("risk_score", ascending=False)
            selected_tx = list(members["TransactionID"].head(MAX_SUBGRAPH_NODES // 2))
            nodes = set(selected_tx)
            for tx in selected_tx:
                if tx in G:
                    nodes.update(G.neighbors(tx))
                if len(nodes) >= MAX_SUBGRAPH_NODES:
                    break
            nodes = set(list(nodes)[:MAX_SUBGRAPH_NODES])
            sub = G.subgraph(nodes).copy()
            truncated = len(members) > len(selected_tx)

            st.caption(
                f"Community {selected_cluster}: {len(members)} transactions total, "
                f"historical fraud rate {members['isFraud'].mean():.1%}. "
                f"Showing a subgraph with the {len(selected_tx)} highest-risk transactions and their direct neighbours."
            )
            show_graph(sub, truncated=truncated)
        else:
            st.info("No community under the chosen threshold: raise the maximum size.")

# ---------------------------------------------------------------------------
# Page: Report Suspicious Activity (dynamic reporting, PDF sec. 8-11)
# ---------------------------------------------------------------------------
elif page == "Report Suspicious Activity":
    st.header("🚨 Report Suspicious Activity")
    st.markdown(
        "A report is **evidence**, not absolute proof (PDF sec. 8): it gets linked to the matching "
        "node (creating it if it doesn't exist yet), the node's risk is recalculated and propagated "
        "to its neighbours with decay — visible right away below and on the *Live Updates* page."
    )

    with st.form("fraud_report_form"):
        entity_choice = st.selectbox("What do you want to report?", list(kr.REPORTABLE_ENTITIES.keys()))
        value = st.text_input("Value (TransactionID, card number, email domain, phone/URL/bank account...)")
        reason = st.selectbox("Type/strength of evidence", list(kr.REPORT_CONFIDENCE_WEIGHTS.keys()))
        notes = st.text_area("Analyst comments / observations")
        submitted = st.form_submit_button("Submit report to the graph engine")

    if submitted:
        if not value:
            st.error("Please enter at least one value to report.")
        else:
            node_id, err = apply_report(entity_choice, value, reason, notes)
            if err:
                st.error(err)
            else:
                st.success(f"Report recorded and linked to node `{node_id}`.")
                last_events = st.session_state.events[: len(st.session_state.events)]
                # only show the events generated by this submit
                just_now = [e for e in st.session_state.events if e["time"] == last_events[0]["time"]]
                for ev in just_now:
                    if ev["type"] == "NEW_NODE":
                        st.info(f"🆕 **NEW NODE** — created `{ev['node']}` ({NODE_TYPE_LABELS.get(ev['node_type'], ev['node_type'])})")
                    elif ev["type"] == "REPORT_ADDED":
                        st.info(f"📝 **REPORT ADDED** — `{ev['report_id']}` linked to `{ev['node']}` (confidence {ev['confidence']:.1f})")
                    elif ev["type"] == "RISK_UPDATED":
                        via = " (propagated)" if ev.get("via_propagation") else ""
                        st.warning(f"📈 **RISK UPDATED**{via} — `{ev['node']}`: {ev['before']:.2f} → {ev['after']:.2f}")
                    elif ev["type"] == "CLASS_CHANGED":
                        via = " (propagated)" if ev.get("via_propagation") else ""
                        st.error(f"🔁 **CLASS CHANGED**{via} — `{ev['node']}`: {ev['before']} → {ev['after']}")

                if st.button("🔗 View the updated node in the Graph Explorer"):
                    st.session_state["ge_prefill_type"] = entity_choice
                    st.session_state["ge_prefill_value"] = value
                    st.session_state.nav_page = "Graph Explorer"
                    st.rerun()

# ---------------------------------------------------------------------------
# Page: Live Updates (PDF sec. 13)
# ---------------------------------------------------------------------------
elif page == "Live Updates":
    st.header("📡 Live Updates")
    st.caption("Full history (most recent first) of every NEW NODE / RISK UPDATED / CLASS CHANGED generated by this session's reports.")

    if st.button("🗑️ Clear log"):
        st.session_state.events = []
        st.rerun()

    if not st.session_state.events:
        st.info("No events yet. Submit a report on the 'Report Suspicious Activity' page to see them appear here.")
    else:
        for ev in st.session_state.events:
            via = " · propagated" if ev.get("via_propagation") else ""
            if ev["type"] == "NEW_NODE":
                st.markdown(f"🆕 `{ev['time']}` **NEW NODE** — `{ev['node']}` ({NODE_TYPE_LABELS.get(ev['node_type'], ev['node_type'])})")
            elif ev["type"] == "REPORT_ADDED":
                st.markdown(f"📝 `{ev['time']}` **REPORT ADDED** — `{ev['report_id']}` → `{ev['node']}` (confidence {ev['confidence']:.1f})")
            elif ev["type"] == "RISK_UPDATED":
                st.markdown(f"📈 `{ev['time']}` **RISK UPDATED**{via} — `{ev['node']}`: {ev['before']:.2f} → {ev['after']:.2f}")
            elif ev["type"] == "CLASS_CHANGED":
                st.markdown(f"🔁 `{ev['time']}` **CLASS CHANGED**{via} — `{ev['node']}`: {ev['before']} → {ev['after']}")
