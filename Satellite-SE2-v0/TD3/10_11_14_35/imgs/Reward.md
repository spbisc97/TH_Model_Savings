```{python}
    def _reward_function(self) -> np.float32:
        ch_radius = self.chaser.radius()
        ch_control = self.chaser.get_control()
        ch_speed = self.chaser.speed()
        ch_state = self.chaser.get_state()

        if self.termination():
            self.terminated = True
            if self.unconstrained or self.success():
                self.is_success = True
                return np.float32(
                    1000
                    + (1_000_000 / ((self.time_step + 1) * self.__step))
                    * (1 / (ch_speed + 1e-4))
                )
            if self.crash():
                return np.float32(-5000 * (ch_speed + 1e-4))

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
        reward_decrease_distance = (reward_decrease_distance * 1e5) / (
            self.xy_max
        )

        reward_distance = -ch_radius / (self.xy_max)

        # Encourage the agent to minimize control effort, with normalization
        reward_control = -np.linalg.norm(ch_control) / (FTMAX)

        # Encourage the agent to maintain low speed
        reward_speed = -ch_speed / (VTRANS_MAX)

        # penalize high angular velocity
        reward_angular_velocity = -np.abs(ch_state[5]) / (VROT_MAX)

        # i could add a reward for the angle between the chaser and the target
        # i coudl add reward_weights to init function

        # Combine
        reward = (
            (self.reward_weights[0] * reward_decrease_distance)
            + (self.reward_weights[1] * reward_distance)
            + (self.reward_weights[2] * reward_control)
            + (self.reward_weights[3] * reward_speed)
            + (self.reward_weights[4] * reward_angular_velocity)
        )
        # print(
        #     reward_decrease_distance,
        #     reward_distance,
        #     reward_control,
        #     reward_speed,
        #     reward_angular_velocity,
        # )
        # print(
        #     (self.reward_weights[0] * reward_decrease_distance),
        #     (self.reward_weights[1] * reward_distance),
        #     (self.reward_weights[2] * reward_control),
        #     (self.reward_weights[3] * reward_speed),
        #     (self.reward_weights[4] * reward_angular_velocity),
        # )

        return np.float32(reward)

```
