from absl import app
from absl import flags

import dm_env
import bsuite

import plotnine as gg
from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis

from bsuite.baselines.experiment import run as benchmark

from rl.agents.q_learning import QLearning

gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'bandit/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Number of episodes to run for.')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')


env = bsuite.load_and_record_to_csv("deep_sea/0", "debug", True)
agent = QLearning(env)

benchmark(agent,
          env,
          1000,
          True)
