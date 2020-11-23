import functools
import jax
import jax.numpy as jnp
import dm_env
from bsuite.baselines import base
from rl.base.q_function import QTable


class QLearning(base.Agent):
    def __init__(self, env, seed=0, alpha=0.01, gamma=1., epsilon=1e-3):
        self.observation_spec = env.observation_spec()
        self.action_spec = env.action_spec()
        self.rng = jax.random.PRNGKey(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = QTable()
        return

    def select_action(
        self,
        timestep: dm_env.TimeStep,
    ) -> base.Action:
        """Selects an action based on an epsilon-greedy policy"""
        # return random action with epsilon probability
        if jax.random.normal(self.rng, (1,)) > self.epsilon:
            return jax.random.randint(self.rng, (1,), 0, self.action_spec.num_values)
        # otherwise compute the q-values for all the available actions on state s_{t+1}
        s = timestep.observation
        q_values = jnp.array([self.q_table.get(s, action) for action in range(self.action_spec.num_values)])
        return int(jnp.argmax(q_values))


    def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
    ):
        """Makes the current policy epsilon-greedy with respect to the current Q-function and updates the Q-table"""
        s = timestep.observation
        r = timestep.reward
        a = action
        s_next = new_timestep.observation

        if r is None:
            return

        # get current estimate
        current_estimate = self.q_table.get(s, a)

        q_values = jnp.array([self.q_table.get(s_next, action) for action in range(self.action_spec.num_values)])
        q = current_estimate + self.alpha * (r + self.gamma * jnp.max(q_values) - current_estimate)
        self.q_table.update(s, a, q)
        return
