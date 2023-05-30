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

        action_noise=action_noise,
        train_freq=(2, "episode"),
        verbose=1,
        gamma=0.999,
        learning_starts=200,

        env = gym.wrappers.TimeLimit(env, max_episode_steps=4000)

