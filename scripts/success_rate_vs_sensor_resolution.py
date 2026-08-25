import matplotlib.pyplot as plt

# Data
rays = [8, 16, 32]
success_rates = [50.8, 44.2, 36.0]

# Figure
plt.figure(figsize=(8, 5))

plt.plot(
    rays,
    success_rates,
    marker='o',
    linewidth=2
)

# Labels
plt.title("Success Rate vs Sensor Resolution", fontsize=16)
plt.xlabel("Number of LiDAR Rays", fontsize=12)
plt.ylabel("Success Rate (%)", fontsize=12)

# Ticks
plt.xticks(rays)
plt.ylim(30, 55)

# Value labels
for x, y in zip(rays, success_rates):
    plt.text(x, y + 0.7, f"{y:.1f}%", ha="center")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "success_rate_vs_sensor_resolution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()