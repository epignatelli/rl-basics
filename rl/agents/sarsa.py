import functools
import jax
import jax.numpy as jnp
import dm_env
from bsuite.baselines import base
from rl.base.q_function import Table


class Sarsa(base.Agent):
    def __init__(self, env, seed=0, alpha=0.01, gamma=1., epsilon=1e-3):
        self.observation_spec = env.observation_spec()
        self.action_spec = env.action_spec()
        self.rng = jax.random.PRNGKey(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = Table()
        return

    def select_action(
        self,
        timestep: dm_env.TimeStep,
    ) -> base.Action:
        """Maps the current state-action pair to a new state-action pair.
        Q-Learning selects an action based on an epsilon-greedy policy"""
        # return random action with epsilon probability
        if jax.random.uniform(self.rng, (1,)) > self.epsilon:
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
        """Makes the current policy epsilon-greedy with respect to
        the current Q-function and updates the Q-table"""
        s0 = timestep.observation
        a0 = action
        r1 = timestep.reward
        s1 = new_timestep.observation
        a1 = self.select_action(new_timestep)

        # get current q-value for state-action s0a0
        q_s0a0 = self.q_table.get(s0, a0)
        # get current q-value for state-action s1a1
        q_s1a1 = self.q_table.get(s1, a1)
        # apply sarsa bellman optimality equation using the action following the current policy
        q = q_s0a0 + self.alpha * (r1 + self.gamma * q_s1a1 - q_s0a0)
        # update the q-value for (s, a)
        self.q_table.update(s0, a0, q)
        return
