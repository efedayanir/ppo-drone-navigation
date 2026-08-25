import matplotlib.pyplot as plt
import numpy as np

# Data
outcomes = ["Success", "Collision", "Timeout", "Stuck"]
baseline = [35.1, 7.7, 48.2, 8.9]
redesign = [44.2, 14.3, 7.2, 34.2]

x = np.arange(len(outcomes))
width = 0.36

plt.figure(figsize=(9, 5.5))

bars1 = plt.bar(
    x - width / 2,
    baseline,
    width,
    label="Baseline PPO"
)

bars2 = plt.bar(
    x + width / 2,
    redesign,
    width,
    label="Reward Redesign"
)

# Titles and labels
plt.title("Failure Mode Shift After Reward Redesign", fontsize=16)
plt.ylabel("Episode Rate (%)", fontsize=12)
plt.xticks(x, outcomes)
plt.ylim(0, 60)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "failure_mode_shift_reward_redesign.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()