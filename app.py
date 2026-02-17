import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(layout="wide")

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_points():
    return pd.read_csv("points.csv")

@st.cache_data
def load_waves_for_feature(feature: str):
    # expects files like waves_passing.csv, waves_receiving.csv, ...
    return pd.read_csv(f"waves_{feature}.csv")

pts_df = load_points()

# Features available from points.csv
FEATURES = sorted(pts_df["feature"].dropna().unique().tolist())

# -----------------------
# Pitch drawing (full pitch, StatsBomb coords)
# -----------------------
def draw_full_pitch(ax, color="white", linewidth=1):
    ax.set_facecolor("#2e7d32")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    # boundaries
    ax.plot([0, 120], [0, 0], color=color, linewidth=linewidth)
    ax.plot([0, 120], [80, 80], color=color, linewidth=linewidth)
    ax.plot([0, 0], [0, 80], color=color, linewidth=linewidth)
    ax.plot([120, 120], [0, 80], color=color, linewidth=linewidth)

    # halfway
    ax.plot([60, 60], [0, 80], color=color, linewidth=linewidth)

    # center circle + spot
    circle = plt.Circle((60, 40), 10, color=color, fill=False, linewidth=linewidth)
    ax.add_patch(circle)
    ax.scatter([60], [40], color=color, s=12, zorder=5)

    # penalty areas
    ax.plot([0, 18], [18, 18], color=color, linewidth=linewidth)
    ax.plot([18, 18], [18, 62], color=color, linewidth=linewidth)
    ax.plot([18, 0], [62, 62], color=color, linewidth=linewidth)

    ax.plot([120, 102], [18, 18], color=color, linewidth=linewidth)
    ax.plot([102, 102], [18, 62], color=color, linewidth=linewidth)
    ax.plot([102, 120], [62, 62], color=color, linewidth=linewidth)

    # 6-yard
    ax.plot([0, 6], [30, 30], color=color, linewidth=linewidth)
    ax.plot([6, 6], [30, 50], color=color, linewidth=linewidth)
    ax.plot([6, 0], [50, 50], color=color, linewidth=linewidth)

    ax.plot([120, 114], [30, 30], color=color, linewidth=linewidth)
    ax.plot([114, 114], [30, 50], color=color, linewidth=linewidth)
    ax.plot([114, 120], [50, 50], color=color, linewidth=linewidth)

    # penalty spots
    ax.scatter([12], [40], color=color, s=12, zorder=5)
    ax.scatter([108], [40], color=color, s=12, zorder=5)

def plot_player_heatmap(points_df, player, feature="passing", bins_x=40, bins_y=30, cmap="hot"):
    sub = points_df[(points_df["player"] == player) & (points_df["feature"] == feature)]
    pts = sub[["x", "y"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    draw_full_pitch(ax)

    if len(pts) == 0:
        ax.set_title(f"{player}\n(no {feature} points)")
        return fig

    H, _, _ = np.histogram2d(
        pts[:,0], pts[:,1],
        bins=[bins_x, bins_y],
        range=[[0,120],[0,80]]
    )
    if H.sum() > 0:
        H = H / H.sum()

    ax.imshow(
        H.T,
        origin="upper",          # y=0 at top (football-view)
        extent=[0,120,0,80],
        cmap=cmap,
        alpha=0.75,
        aspect="equal"
    )

    ax.set_title(f"{player} — {feature}")
    return fig

# -----------------------
# Similarity (Cosine on serpentine wave)
# -----------------------
def get_wave_cols(waves_df):
    return [c for c in waves_df.columns if c.startswith("wave_")]

def get_wave_matrix(df, wave_cols):
    return df[wave_cols].to_numpy(dtype=float)

@st.cache_data
def cosine_topk(waves_df, target_player, k=3):
    df = waves_df.reset_index(drop=True)
    wave_cols = get_wave_cols(df)
    V = get_wave_matrix(df, wave_cols)
    idx = {p:i for i,p in enumerate(df["player"].tolist())}

    if target_player not in idx:
        raise ValueError("Target player not found in waves file for this feature")

    t = idx[target_player]
    d = cosine_distances(V[t:t+1], V).ravel()
    order = np.argsort(d)

    out = []
    for j in order:
        if df.loc[j, "player"] == target_player:
            continue
        out.append((df.loc[j, "player"], float(d[j])))
        if len(out) == k:
            break
    return out

def get_player_wave(waves_df, player):
    wave_cols = get_wave_cols(waves_df)
    row = waves_df[waves_df["player"] == player].iloc[0]
    return row[wave_cols].to_numpy(dtype=float)

def plot_wave_comparison(waves_df, target, sim_list, title=""):
    fig, ax = plt.subplots(figsize=(14, 4))

    wt = get_player_wave(waves_df, target)
    ax.plot(wt, label=f"TARGET: {target}", linewidth=2)

    for p, dist in sim_list:
        w = get_player_wave(waves_df, p)
        ax.plot(w, label=f"{p} (cos={dist:.3f})", alpha=0.85)

    ax.set_title(title or "Serpentine wave comparison")
    ax.set_xlabel("Wave index (pseudo-time)")
    ax.set_ylabel("Density")
    ax.legend()
    return fig

# -----------------------
# UI
# -----------------------
st.title("Player Similarity — Serpentine Wave")

# Choose feature
feature = st.selectbox("Select feature to compare", FEATURES, index=FEATURES.index("passing") if "passing" in FEATURES else 0)

# Load waves for selected feature
try:
    waves_df = load_waves_for_feature(feature)
except FileNotFoundError:
    st.error(f"Missing file: waves_{feature}.csv\n\nExport it from the notebook first.")
    st.stop()

players = sorted(waves_df["player"].unique().tolist())
target = st.selectbox("Select target player", players)

k = st.slider("Top-K similar players", 1, 10, 3, 1)

topk = cosine_topk(waves_df, target, k=k)

st.subheader(f"Top similar players (Cosine on serpentine wave) — feature: {feature}")
st.write(topk)

st.divider()

# Wave comparison
st.header("Wave comparison")
fig_wave = plot_wave_comparison(waves_df, target, topk, title=f"Serpentine wave — {feature}")
st.pyplot(fig_wave)

st.divider()

# Heatmaps section
st.header(f"Heatmaps (Full Pitch) — {feature}")
st.caption("Showing target + Top-K similar players (Cosine).")

def show_heatmaps_block(title, names):
    st.subheader(title)
    cols = st.columns(len(names))
    for c, name in zip(cols, names):
        with c:
            fig = plot_player_heatmap(pts_df, name, feature=feature, cmap="hot")
            st.pyplot(fig)

show_heatmaps_block("Target + Similar Players", [target] + [p for p,_ in topk])
