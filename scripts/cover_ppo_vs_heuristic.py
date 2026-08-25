import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# ============================================================
# Data
# ============================================================

baseline = 35.1
reward_redesign = 44.2
best_ppo = 50.8
heuristic = 84.3

output_file = "medium_cover_strong_story_graphic.png"

# ============================================================
# Canvas
# ============================================================

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ============================================================
# Subtle background trajectory lines
# ============================================================

t = np.linspace(0, 1, 300)

# PPO messy path
x = 8 + 35 * t
y = 28 + 10 * np.sin(10 * np.pi * t) + 5 * np.sin(27 * np.pi * t)
ax.plot(x, y, color="#ef4444", alpha=0.16, linewidth=5)

# Heuristic clean path
x2 = 58 + 32 * t
y2 = 25 + 30 * t
ax.plot(x2, y2, color="#22c55e", alpha=0.18, linewidth=6)

# Decorative obstacle dots
obstacles = [
    (18, 65, 3.8),
    (31, 52, 5.2),
    (68, 33, 4.4),
    (80, 65, 5.0),
    (46, 36, 3.2),
]

for ox, oy, r in obstacles:
    ax.add_patch(
        Circle(
            (ox, oy),
            r,
            facecolor="#334155",
            edgecolor="#64748b",
            linewidth=1.5,
            alpha=0.55
        )
    )

# ============================================================
# Header
# ============================================================

ax.text(
    50,
    91,
    "I Improved My PPO Agent.",
    ha="center",
    fontsize=38,
    fontweight="bold",
    color="white"
)

ax.text(
    50,
    84,
    "It Still Lost to Simple Rules.",
    ha="center",
    fontsize=34,
    fontweight="bold",
    color="#e2e8f0"
)

# ============================================================
# Main comparison cards
# ============================================================

# PPO card
ppo_card = FancyBboxPatch(
    (8, 35),
    36,
    34,
    boxstyle="round,pad=0.8,rounding_size=3",
    facecolor="#1e293b",
    edgecolor="#ef4444",
    linewidth=3
)
ax.add_patch(ppo_card)

ax.text(
    26,
    62,
    "BEST PPO",
    ha="center",
    fontsize=22,
    fontweight="bold",
    color="#fca5a5"
)

ax.text(
    26,
    50,
    f"{best_ppo:.1f}%",
    ha="center",
    fontsize=58,
    fontweight="bold",
    color="#ef4444"
)

ax.text(
    26,
    41,
    "after reward redesign\nand sensor reduction",
    ha="center",
    fontsize=15,
    color="#cbd5e1"
)

# Heuristic card
heur_card = FancyBboxPatch(
    (56, 35),
    36,
    34,
    boxstyle="round,pad=0.8,rounding_size=3",
    facecolor="#1e293b",
    edgecolor="#22c55e",
    linewidth=3
)
ax.add_patch(heur_card)

ax.text(
    74,
    62,
    "SIMPLE HEURISTIC",
    ha="center",
    fontsize=22,
    fontweight="bold",
    color="#86efac"
)

ax.text(
    74,
    50,
    f"{heuristic:.1f}%",
    ha="center",
    fontsize=58,
    fontweight="bold",
    color="#22c55e"
)

ax.text(
    74,
    41,
    "hand-crafted rules\nstill performed better",
    ha="center",
    fontsize=15,
    color="#cbd5e1"
)

# VS marker
ax.text(
    50,
    52,
    "VS",
    ha="center",
    va="center",
    fontsize=32,
    fontweight="bold",
    color="white"
)

# ============================================================
# PPO improvement strip
# ============================================================

strip = FancyBboxPatch(
    (15, 17),
    70,
    9,
    boxstyle="round,pad=0.45,rounding_size=2",
    facecolor="#020617",
    edgecolor="#334155",
    linewidth=2
)
ax.add_patch(strip)

ax.text(
    50,
    28.5,
    "PPO improved, but it did not win",
    ha="center",
    fontsize=18,
    fontweight="bold",
    color="#e2e8f0"
)

ax.text(
    50,
    21.2,
    f"{baseline:.1f}%   →   {reward_redesign:.1f}%   →   {best_ppo:.1f}%",
    ha="center",
    fontsize=26,
    fontweight="bold",
    color="#f87171"
)

ax.text(
    23,
    14,
    "Baseline",
    ha="center",
    fontsize=12,
    color="#94a3b8"
)

ax.text(
    50,
    14,
    "Reward redesign",
    ha="center",
    fontsize=12,
    color="#94a3b8"
)

ax.text(
    77,
    14,
    "Best PPO",
    ha="center",
    fontsize=12,
    color="#94a3b8"
)

# ============================================================
# Footer
# ============================================================

ax.text(
    50,
    6,
    "Learned policy improved. Simple navigation rules still won.",
    ha="center",
    fontsize=17,
    color="#cbd5e1"
)

# ============================================================
# Export
# ============================================================

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()