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
            reward_decreased_distance = (
                np.linalg.norm(self.state_history[-1][0:2])
            ) - ch_radius
        else:
            reward_decreased_distance = 0
        reward_decreased_distance = reward_decreased_distance * (
            self.xy_max / (ch_radius + 1e-4)
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
            (self.reward_weights[0] * reward_decreased_distance)
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
```{python}
{'O_params': {'dt': 0.01, 'initial_noise': None, 'theta': 0.2},
 'env_params': {'initial_integration_steps': array([  0, 400], dtype=int32),
                'reward_weights': array([20. ,  0.8,  0.5,  1. , 30. ], dtype=float32),
                'starting_noise': array([5.0000000e+00, 9.9999998e-03, 6.2831855e+00, 1.0000000e-03,
       0.0000000e+00, 0.0000000e+00], dtype=float32),
                'starting_state': array([10.,  0.,  0.,  0.,  0.,  0.], dtype=float32),
                'step': 0.1,
                'underactuated': True},
 'params': {'dtype': <class 'numpy.float32'>,
            'mean': array([0., 0.]),
            'sigma': array([1.e-03, 1.e-05], dtype=float32)},
 'params_algo': {'batch_size': 256,
                 'ent_coef': 0.01,
                 'env': <stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv object at 0x7fdad37ab9d0>,
                 'gae_lambda': 0.95,
                 'gamma': 0.9997,
                 'learning_rate': 0.0002,
                 'n_epochs': 10,
                 'n_steps': 4096,
                 'policy': 'MlpPolicy',
                 'policy_kwargs': {'net_arch': [256, 512, 256]},
                 'stats_window_size': 30,
                 'tensorboard_log': 'savings/Satellite-SE2-v0/PPO/10_13_10_17/logs/',
                 'use_sde': True,
                 'verbose': 1},
 'params_learn': {'log_interval': 2,
                  'progress_bar': False,
                  'reset_num_timesteps': False,
                  'total_timesteps': 200000}}
```
