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
# comments

for the first 4 * 50_000 steps no action noise was used


the othe parameters were the default ones
 learning_rate=0.0001, buffer_size=1000000, learning_starts=50000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1
 target_update_interval=10000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05