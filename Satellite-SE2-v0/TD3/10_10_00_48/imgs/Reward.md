```{python}
    def _reward_function(self) -> np.float32:
        ch_radius = self.chaser.radius()
        ch_control = self.chaser.get_control()
        ch_speed = self.chaser.speed()
        ch_state = self.chaser.get_state()
        reward_weights = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        if self.termination():
            self.terminated = True
            if self.unconstrained or self.success():
                return np.float32(
                    1000 + 
                    (1_000_000 / (self.time_step * self.__step)) * 
                    (1 / ch_speed))
            if self.crash():
                return np.float32(-5000*ch_speed)

        if self.out_of_bounds():
            self.truncated = True
            return np.float32(-1000)

        # for shaping i could add a reward for the radius lowering

        # Encourage the agent to minimize the distance
        if self.time_step != 0:
            reward_decrease_distance = (
                np.linalg.norm(self.state_history[-1][0:2])
            ) - ch_radius
        else:
            reward_decrease_distance = 0
        reward_weights[0] = 10

        reward_distance = -ch_radius / (self.xy_max)
        reward_weights[1] = 1

        # Encourage the agent to minimize control effort, with normalization
        reward_control = -np.linalg.norm(ch_control) / (FTMAX)
        reward_weights[2] = 0.05

        # Encourage the agent to maintain low speed
        reward_speed = -ch_speed / (VTRANS_MAX)
        reward_weights[3] = 0.1

        # penalize high angular velocity
        reward_angular_velocity = -np.abs(ch_state[5]) / (VROT_MAX)
        reward_weights[4] = 10

        # i could add a reward for the angle between the chaser and the target
        # i coudl add reward_weights to init function

        # Combine
        reward = (
            (reward_weights[0] * reward_decrease_distance)
            + (reward_weights[1] * reward_distance)
            + (reward_weights[2] * reward_control)
            + (reward_weights[3] * reward_speed)
            + (reward_weights[4] * reward_angular_velocity)
        )

        return np.float32(reward)

```


```
Use of a bigger net [400,300]
```