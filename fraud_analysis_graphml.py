import kagglehub
import pandas as pd
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import networkx as nx
import scipy
import seaborn as sns
from infomap import Infomap
from sklearn.model_selection import train_test_split
from collections import deque, Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from matplotlib import colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def draw_network(
    df,
    source_col,
    target_col,
    layout="random",  # "random", "spring", "kamada_kawai"
    sample=1_000,
    seed=1,
    node_size=12,
):
    # Sample for responsiveness
    df_plot = df.sample(n=min(sample, len(df)), random_state=seed)

    # Build graph
    G = nx.from_pandas_edgelist(df_plot, source=source_col, target=target_col)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout in ("kamada_kawai", "kk"):
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G, seed=seed)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#010203"),
        hoverinfo="none",
        mode="lines",
        name="edges",
    )

    # Nodes (color by degree)
    degrees = dict(G.degree())
    node_x, node_y, node_color, node_text = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        d = degrees.get(n, 0)
        node_color.append(d)
        node_text.append(f"Node: {n}<br>Degree: {d}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="RdBu",
            reversescale=False,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=35,
                title=dict(text="Node Connections", side="right"),  # fixed
                xanchor="left",
            ),
            line=dict(width=2),
        ),
        name="nodes",
    )

    # Figure (use title as dict; no titlefont)
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="<b>Network Graph of Provider & Physician</b>",
                font=dict(size=16),
                x=0.5,
                xanchor="center",
            ),
            showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[],  # removed the "new text"
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


if __name__ == "__main__":

    path = kagglehub.dataset_download(
        "rohitrox/healthcare-provider-fraud-detection-analysis"
    )

    print("Path to dataset files:", path)

    downloaded_path = "~/.cache/kagglehub/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/versions/1/"

    train_df = pd.read_csv(os.path.join(downloaded_path, "Train-1542865627584.csv"))
    train_beneficiary_df = pd.read_csv(
        os.path.join(downloaded_path, "Train_Beneficiarydata-1542865627584.csv")
    )
    train_inpatient_df = pd.read_csv(
        os.path.join(downloaded_path, "Train_Inpatientdata-1542865627584.csv")
    )
    train_outpatient_df = pd.read_csv(
        os.path.join(downloaded_path, "Train_Outpatientdata-1542865627584.csv")
    )

    df1 = train_inpatient_df[
        [
            "BeneID",
            "ClaimID",
            "Provider",
            "InscClaimAmtReimbursed",
            "AttendingPhysician",
            "DeductibleAmtPaid",
        ]
    ]
    df2 = train_outpatient_df[
        [
            "BeneID",
            "ClaimID",
            "Provider",
            "InscClaimAmtReimbursed",
            "AttendingPhysician",
            "DeductibleAmtPaid",
        ]
    ]
    df3 = train_beneficiary_df[
        [
            "BeneID",
            "Gender",
            "Race",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
        ]
    ]

    df = pd.concat([df1, df2])

    df = df.merge(train_df, on="Provider", how="inner").merge(
        df3, on="BeneID", how="inner"
    )

    df["PotentialFraud"] = (
        df["PotentialFraud"].replace("No", 0).replace("Yes", 1).astype(int)
    )
    df["Provider"] = (
        df["Provider"].str.removeprefix("PRV").fillna(0).astype("int") + 100000
    )
    df["AttendingPhysician"] = (
        df["AttendingPhysician"].str.removeprefix("PHY").fillna(0).astype("int")
        + 200000
    )

    source = "Provider"
    target = "AttendingPhysician"
    G = nx.from_pandas_edgelist(df, source=source, target=target)
    # draw_network(df, source_col=source, target_col=target, layout="spring", sample=1000, seed=42)
    nodes_info_dict = {
        "eigenvector_centrality": nx.eigenvector_centrality(G),
        "pagerank": nx.pagerank(G),
    }
    columns_with_node_infos = ["degree"] + list(nodes_info_dict.keys())

    nodes_info = (
        pd.DataFrame.from_dict(dict(nx.degree(G)), orient="index")
        .rename(columns={0: "degree"})
        .reset_index()
        .rename(columns={"index": "node"})
    )

    # computing graph features for each node
    metrics = (
        pd.DataFrame(nodes_info_dict).reset_index().rename(columns={"index": "node"})
    )
    node_info = nodes_info.merge(metrics, on="node", how="left")
    node_info.rename(columns={"node": "Physician"}, inplace=True)

    df_enriched = df.merge(
        node_info, left_on="Provider", right_on="Physician", how="left"
    ).drop("Physician", axis=1)

    df_enriched.rename(
        columns={k: "Provider_" + k for k in columns_with_node_infos}, inplace=True
    )

    df_enriched = df_enriched.merge(
        node_info, left_on="AttendingPhysician", right_on="Physician", how="left"
    ).drop("Physician", axis=1)

    df_enriched.rename(
        columns={k: "AttendingPhysician_" + k for k in columns_with_node_infos},
        inplace=True,
    )

    corr_data = df_enriched[
        [
            "PotentialFraud",
            "AttendingPhysician",
            "Provider_degree",
            "Provider_eigenvector_centrality",
            "Provider_pagerank",
            "AttendingPhysician_degree",
            "AttendingPhysician_eigenvector_centrality",
            "AttendingPhysician_pagerank",
        ]
    ]
    corr = corr_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        annot=True,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.tight_layout()

    plt.show()

    def findCommunities(G):
        im = Infomap(two_level=False, silent=True)
        for u, v in G.edges():
            im.addLink(u, v)  # or im.add_link(u, v) depending on your binding
        im.run()

        # node -> community (flat)
        comm = {}
        for node in im.iterTree():
            if node.isLeaf():
                comm[node.node_id] = node.module_id

        nx.set_node_attributes(G, comm, "community")
        print(f"modules: {len(set(comm.values()))}")
        return G

    def _largest_components(G, top_k_components=1, min_component_size=None):
        if G.is_directed():
            comps = list(nx.weakly_connected_components(G))
        else:
            comps = list(nx.connected_components(G))
        comps = sorted(comps, key=len, reverse=True)
        if min_component_size is not None:
            keep = [c for c in comps if len(c) >= min_component_size] or (
                [comps[0]] if comps else []
            )
        else:
            keep = comps[: max(1, int(top_k_components))] if comps else []
        return keep

    def _bfs_connected_sample(H, target_nodes, seed=1):
        """Return an induced subgraph of ~target_nodes that remains connected."""
        rng = random.Random(seed)
        if H.number_of_nodes() <= target_nodes:
            return H.copy()

        # Start from a high-degree node for denser neighborhoods
        start = max(H.degree, key=lambda x: x[1])[0]
        visited, q = {start}, deque([start])

        while q and len(visited) < target_nodes:
            u = q.popleft()
            nbrs = list(H.neighbors(u))
            rng.shuffle(nbrs)
            for v in nbrs:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                if len(visited) >= target_nodes:
                    break

        S = H.subgraph(visited).copy()
        # ensure connected (keep giant component of sampled nodes)
        if H.is_directed():
            comp = max(nx.weakly_connected_components(S), key=len)
        else:
            comp = max(nx.connected_components(S), key=len)
        return S.subgraph(comp).copy()

    def drawNetwork_sampled_connected(
        G,
        top_k_components=3,  # keep K largest components (before sampling)
        min_component_size=None,  # or keep comps with size >= this (overrides K)
        max_nodes=1500,  # target nodes to draw (connected BFS sample)
        max_edges=30000,  # optional hard cap on edges
        layout="spring",
        seed=1,
    ):
        # ---- pick largest components, then union them ----
        keep = _largest_components(G, top_k_components, min_component_size)
        keep_nodes = set().union(*keep) if keep else set(G.nodes())
        H = G.subgraph(keep_nodes).copy()

        # ---- connected sampling (BFS) so edges don't evaporate ----
        H = _bfs_connected_sample(H, target_nodes=max_nodes, seed=seed)

        # optional edge cap
        if H.number_of_edges() > max_edges:
            rng = random.Random(seed)
            edges = rng.sample(list(H.edges()), max_edges)
            H2 = nx.DiGraph() if G.is_directed() else nx.Graph()
            H2.add_nodes_from(H.nodes(data=True))
            H2.add_edges_from(edges)
            H = H2

        # ---- layout ----
        if layout == "spring":
            k = 1 / np.sqrt(max(H.number_of_nodes(), 1))
            pos = nx.spring_layout(H, seed=seed, k=k, iterations=50)
        else:
            pos = nx.random_layout(H, seed=seed)

        # ---- community colors aligned to node order ----
        comm_attr = nx.get_node_attributes(H, "community")
        nodes = list(H.nodes())
        mods_raw = [comm_attr.get(n, -1) for n in nodes]  # -1 for unlabeled
        uniq = sorted(set(mods_raw))
        id_map = {m: i for i, m in enumerate(uniq)}
        mods = [id_map[m] for m in mods_raw]
        K = len(uniq)

        base_light = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
        base_dark = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"]
        reps = int(np.ceil(max(1, K / len(base_light))))
        cmapLight = colors.ListedColormap((base_light * reps)[:K], name="light")
        cmapDark = colors.ListedColormap((base_dark * reps)[:K], name="dark")

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.set_axis_off()
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.4)

        vmin, vmax = -0.5, K - 0.5
        ncoll = nx.draw_networkx_nodes(
            H,
            pos,
            ax=ax,
            node_size=12,
            node_color=mods,
            cmap=cmapLight,
            vmin=vmin,
            vmax=vmax,
        )
        ncoll.set_edgecolor([cmapDark(i % cmapDark.N) for i in mods])
        ncoll.set_linewidth(0.5)

        ax.set_title(
            f"Connected sample from largest comps: {H.number_of_nodes()} nodes, "
            f"{H.number_of_edges()} edges, {K} communities",
            fontsize=12,
        )
        plt.tight_layout()
        # plt.show()

    # usage
    G = findCommunities(G)
    drawNetwork_sampled_connected(G, min_component_size=300, max_nodes=1500)

    df_communities = pd.DataFrame(
        [[k, v] for k, v in nx.get_node_attributes(G, "community").items()],
        columns=["AttendingPhysician", "AttendingPhysician_cluster"],
    )

    df_enriched = (
        df_enriched.set_index("AttendingPhysician")
        .join(
            df_communities.set_index("AttendingPhysician"), how="left", rsuffix="_comm"
        )
        .reset_index()
    )

    X, y = (
        df_enriched[
            [
                "InscClaimAmtReimbursed",
                "DeductibleAmtPaid",
                "Gender",
                "Race",
                "NoOfMonths_PartACov",
                "NoOfMonths_PartBCov",
                "IPAnnualReimbursementAmt",
                "IPAnnualDeductibleAmt",
                "OPAnnualReimbursementAmt",
                "OPAnnualDeductibleAmt",
                "Provider_degree",
                #  'Provider_closeness_centrality',
                "Provider_eigenvector_centrality",
                "Provider_pagerank",
                "AttendingPhysician_degree",
                #  'AttendingPhysician_closeness_centrality',
                "AttendingPhysician_eigenvector_centrality",
                "AttendingPhysician_pagerank",
                "AttendingPhysician_cluster",
            ]
        ],
        df_enriched["PotentialFraud"],
    )
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=69
    )

    print(
        f"Shapes after splitting:\n\nX_train: {X_train.shape}, y_train: {y_train.shape}\
      \nX_test: {X_test.shape}, y_test: {y_test.shape}"
    )

    classes = df_enriched["PotentialFraud"].unique()

    plt.hist(classes)
    plt.title("Imbalance in classes")
    # plt.show()

    feature_names = [
        "Baseline",
        "Graph Features",
        "Graph Features with Community Detection",
    ]

    features = [
        # Baseline features
        [
            "InscClaimAmtReimbursed",
            "DeductibleAmtPaid",
            "Gender",
            "Race",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
        ],
        # Baseline + Graph Features
        [
            "InscClaimAmtReimbursed",
            "DeductibleAmtPaid",
            "Gender",
            "Race",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
            "Provider_degree",
            "Provider_eigenvector_centrality",
            "Provider_pagerank",
            "AttendingPhysician_degree",
            "AttendingPhysician_eigenvector_centrality",
            "AttendingPhysician_pagerank",
        ],
        # Baseline + Graph Features + Community Detection
        [
            "InscClaimAmtReimbursed",
            "DeductibleAmtPaid",
            "Gender",
            "Race",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
            "Provider_degree",
            "Provider_eigenvector_centrality",
            "Provider_pagerank",
            "AttendingPhysician_degree",
            "AttendingPhysician_eigenvector_centrality",
            "AttendingPhysician_pagerank",
            "AttendingPhysician_cluster",
        ],
    ]
    hyper_parameter_grids_RFC = [
        {  # Grid 1: No regularization
            "randomforestclassifier__criterion": ["gini"],
            "randomforestclassifier__max_depth": [10, 20, 50, 100, 250, 300, 500],
            "randomforestclassifier__min_samples_split": [2, 3, 5, 10, 20, 30],
        },
        {  # Grid 2: L2 regularization
            "randomforestclassifier__criterion": ["entropy"],
            "randomforestclassifier__max_depth": [10, 20, 50, 100, 250, 300, 500],
            "randomforestclassifier__min_samples_split": [2, 3, 5, 10, 20, 30],
        },
    ]
    pipeline_RFC = make_pipeline(
        StandardScaler(), RandomForestClassifier(random_state=69)
    )

    accuracy = []
    evaluated_names = []

    for feature in features:
        # ensure 2D arrays for sklearn
        X_train_subset = X_train[feature]
        X_test_subset = X_test[feature]

        clf = RandomizedSearchCV(
            pipeline_RFC,
            hyper_parameter_grids_RFC,
            scoring="accuracy",
            cv=4,
            n_jobs=-1,
            n_iter=10,
            random_state=42,  # optional, for reproducibility
        )
        clf.fit(X_train_subset, y_train)

        acc = round(clf.best_estimator_.score(X_test_subset, y_test) * 100, 2)

        print("\nBest params:", clf.best_params_)
        print("Dev CV best score:", round(clf.best_score_ * 100, 2), "%")
        print(f"Test accuracy for {feature}: {acc}%")

        evaluated_names.append(feature)
        accuracy.append(acc)

    # plot once, after the loop
    custom_labels = [
        "baseline",
        "baseline + graph features",
        "baseline + graph features + community detection",
    ]

    labels = custom_labels[: len(accuracy)]

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, accuracy, color="maroon", width=0.6)
    plt.xticks(x, labels, rotation=0, ha="center")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy by feature set")
    plt.tight_layout()
    plt.show()

    bar_data = pd.DataFrame(
        dict(
            labels=list(X_train.columns),
            feature_importance=clf.best_estimator_.steps[1][1].feature_importances_,
        )
    )

    bar_data = bar_data.sort_values("feature_importance", ascending=False)

    bar_data.plot(x="labels", y="feature_importance", kind="bar")

    plt.tight_layout()
