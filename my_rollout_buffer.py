import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

class LeftShiftPadRightRolloutBuffer(RolloutBuffer):
    """
    For each episode segment in each env column:
      shifted[t] = rewards[t + delay]           (left shift)
    and for the last delay steps:
      shifted[-k:] = last_available_reward      (right padding by repetition)

    Example (delay=2):
      actions: [a_t, a_{t+1}, ..., a_{t+n}]
      rewards: [r_t, r_{t+1}, ..., r_{t+n}]
      used by PPO -> [r_{t+2}, r_{t+3}, ..., r_{t+n-2}, r_{t+n-2}, r_{t+n-2}]
    """
    def __init__(self, *args, delay: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        assert delay >= 1
        self.delay = delay
        print(f"🔧 LeftShiftPadRightRolloutBuffer initialized with delay={self.delay}")


    def compute_returns_and_advantage(self, last_values, dones):
        print(f"🔄 compute_returns_and_advantage called with delay={self.delay}")
        # Save original for comparison
        r_original = self.rewards.copy()
        r = self.rewards.copy()              # (n_steps, n_envs)
        starts = self.episode_starts.copy()  # True at episode starts

        for env_i in range(r.shape[1]):
            idx = 0
            while idx < r.shape[0]:
                # find [seg_start, seg_end) for this episode segment
                seg_start = idx
                idx += 1
                while idx < r.shape[0] and not starts[idx, env_i]:
                    idx += 1
                seg_end = idx

                seg = r[seg_start:seg_end, env_i]  # shape (L,)
                L = len(seg)
                if L == 0:
                    continue

                d = min(self.delay, L)
                shifted = np.empty_like(seg)

                if L > d:
                    shifted[:L - d] = seg[d:]                # left shift
                    last_val = seg[L - d - 1]                # last available after shift
                    shifted[L - d:] = last_val               # pad right by repeating
                else:
                    # entire segment becomes the last available reward
                    last_val = seg[-1]
                    shifted[:] = last_val

                r[seg_start:seg_end, env_i] = shifted

        # DEBUG: Show before/after
        print("="*70)
        print(f"SEGMENT LENGTH: {seg_end - seg_start if 'seg_end' in locals() else 'N/A'}")
        print(f"BEFORE SHIFT (env 0, first 10 steps):\n{r_original[:10, 0]}")
        print(f"AFTER SHIFT  (env 0, first 10 steps):\n{r[:10, 0]}")
        print(f"Expected: r[0] should ≈ r_original[3] (with delay=3)")
        print("="*70)
        self.rewards = r
        # proceed with standard GAE/returns on the shifted rewards
        super().compute_returns_and_advantage(last_values, dones)