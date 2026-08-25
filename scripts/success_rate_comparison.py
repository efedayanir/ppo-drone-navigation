import matplotlib.pyplot as plt

methods = [
    "PPO (best)",
    "Greedy Goal",
    "Obstacle Aware",
    "Wall Avoiding"
]

success = [50.8, 83.0, 79.0, 84.0]

plt.figure(figsize=(8, 5))

bars = plt.bar(methods, success)

plt.title("Success Rate Comparison", fontsize=16)
plt.ylabel("Success Rate (%)", fontsize=12)
plt.ylim(0, 100)

# value labels
for bar, value in zip(bars, success):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "ppo_vs_baselines_success_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()