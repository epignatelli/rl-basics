from typing import Any, Callable, NamedTuple, Tuple
import functools
import jax.numpy as jnp


QFunction = Any
ActionSpace = Tuple[int, ...]
State = jnp.ndarray

class Agent(NamedTuple):
    select_action: Callable[[QFunction, State, ActionSpace, float, int], int]
    update: Callable[[QFunction, State, int, State, float, ActionSpace, float, float], None]


def agent(agent_maker):
    @functools.wraps(agent_maker)
    def fabricate_module(*args, **kwargs):
        select_action, update = agent_maker(*args, **kwargs)
        return Agent(select_action, update)
    return fabricate_module