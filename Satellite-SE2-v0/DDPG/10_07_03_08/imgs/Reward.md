```{python}
    def _reward_function(self):
        reward = 0
        ch_radius = self.chaser.radius()
        ch_control = self.chaser.get_control()
        ch_speed = self.chaser.speed()
        ch_state = self.chaser.get_state()
        w_speed = 1e2

        reward += (
            (-np.log10(ch_radius + 0.1))
            - (np.linalg.norm(ch_control) / (FTMAX * 3))
            - (ch_speed * w_speed)  # chaser abs speed
            - np.linalg.norm(ch_state[5]) * w_speed  # angular velocity
        )
        return reward

```
