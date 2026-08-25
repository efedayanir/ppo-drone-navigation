import matplotlib.pyplot as plt
import numpy as np

# Data
configs = [
    "With Progress\nReward",
    "Without Progress\nReward"
]

success = [50.8, 0.9]
collision = [15.6, 99.1]

x = np.arange(len(configs))
width = 0.35

# Plot
plt.figure(figsize=(8, 5))

bars1 = plt.bar(
    x - width/2,
    success,
    width,
    label="Success Rate"
)

bars2 = plt.bar(
    x + width/2,
    collision,
    width,
    label="Collision Rate"
)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

# Formatting
plt.title("Progress Reward Ablation", fontsize=16)
plt.ylabel("Episode Rate (%)", fontsize=12)
plt.xticks(x, configs)
plt.ylim(0, 110)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "progress_reward_ablation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()