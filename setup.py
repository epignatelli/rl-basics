#!/usr/bin/env python

from distutils.core import setup

setup(name='rl-basics',
      version='0.0.1',
      description='Basic algorithms for reinforcement learning',
      author='Eduardo Pignatelli',
      author_email='edu.pignatelli@gmail.com',
      url='https://github.com/epignateli/rl-basics',
      packages=['rl', 'rl.agents', 'rl.base'],
      install_requires=["bsuite", "jaxlib", "jax"]
     )