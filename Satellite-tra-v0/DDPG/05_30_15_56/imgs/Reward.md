```{python}
    def _reward_function(self, action=0, terminated=False):
        reward = 0
        term_reward = 0 if not terminated else -100_000

        # Shaping Term (Not Used)
        # shaping = self._shape_reward()
        #
        # if self.prev_shaping is not None:
        # reward = shaping - self.prev_shaping
        # self.prev_shaping = shaping

        # Position Error Term
        log_position_error_term = np.log(np.linalg.norm(self.chaser.state[:3]) + 1e-1)

        # Control Effort Term
        control_effort_term = 0.3 * np.linalg.norm(action)

        reward += term_reward - log_position_error_term - control_effort_term

        return float(reward)

```

for the first 3.5e06 steps time limit = 4000
gamma = 0.999
plots are done without determinist policy

since step 3533000 the time limit is 5000
and the episodes are reported with deterministic policy


learning is always each 2 episodes



Maybe there is too much noise added to the function and the agent is not able to learn properly

noise = 0.02