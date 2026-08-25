| ID      | Purpose              | Rays | Reward      | Seeds      | Timesteps | Result |
| ------- | -------------------- | ---: | ----------- | ---------- | --------: | -----: |
| PPO-B16 | Baseline             |   16 | Original    | 42,123,999 |      800k |  35.1% |
| PPO-R16 | Reward redesign      |   16 | Redesigned  | 42,123,999 |      800k |  44.2% |
| PPO-R8  | Sensor reduction     |    8 | Redesigned  | 42,123,999 |      800k |  50.8% |
| PPO-R32 | Sensor increase      |   32 | Redesigned  | 42,123,999 |      800k |  36.0% |
| PPO-NP8 | No-progress ablation |    8 | No progress | 42,123,999 |      800k |   0.9% |
