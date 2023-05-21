```{python}
    def _reward_function(self, action, terminated=False):
        reward = 0
        term_reward = 0 if not terminated else -100_000

        # # Shaping Reward not used rn
        # shaping = self._shape_reward()
        #
        # if self.prev_shaping is not None:
        # reward = shaping - self.prev_shaping
        # self.prev_shaping = shaping

        # Attitude Error Term
        attitude_error_term = -1e0 * np.linalg.norm(
            self.chaser.quat_track_err(self.chaser.state[:4], self.qd)[1:]
        )

        # Control Effort Term
        control_effort_term = -1e-2 * np.sum(np.abs(action))

        # Stability Term
        stability_term = -1e-1 * np.dot(self.chaser.state[4:8], self.chaser.state[4:8])

        # Smoothness Term
        # smoothness_term = -0.001 * np.linalg.norm(np.gradient(angular_velocity))

        # Total Reward
        reward += (
            term_reward
            + attitude_error_term
            + control_effort_term
            + stability_term
            #  + smoothness_term
        )
        return float(reward)

```

# Comments
After ~3_000_000 steps it starts overfitting and gets bad rewards