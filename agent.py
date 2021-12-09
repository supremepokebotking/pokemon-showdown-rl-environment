import tensorflow as tf
import numpy as np
import gym
import math
import os

import model
import architecture as policies
import poke_sim_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
#from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
#from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from tensorflow.keras import backend as K


def main():
	graph_options = tf.GraphOptions(place_pruned_graph =False)
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, graph_options=graph_options)
	config = tf.ConfigProto()
	# Avoid warning message errors
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	# Allowing GPU memory growth
	config.gpu_options.allow_growth = True
	K.clear_session()

	with tf.Session(config=config):
		envs_for_models = [
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
                                env.make_poke_env(),
								]
		full_envs = [
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_priority_attack(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_last_breath(),
			env.make_poke_env_strategic(),
			env.make_poke_env_strategic(),
			env.make_poke_env_strategic(),
			env.make_poke_env_strategic(),
			env.make_poke_env(),
			env.make_poke_env(),
			env.make_poke_env(),
			]
		full_envs.extend(envs_for_models)
		model.learn(policy=policies.PPOPolicy,
							env=SubprocVecEnv(full_envs),
							nsteps=80, # Steps per environment
#							nsteps=2048, # Steps per environment
#							total_timesteps=10000000,
							total_timesteps=10000000,
							gamma=0.99,
							lam=0.95,
							vf_coef=0.5,
							ent_coef=0.01,
							lr = lambda _:2e-4,
							cliprange = lambda _:0.2, # 0.1 * learning_rate
							max_grad_norm = 0.5,
							log_interval  = 10,
							envs_for_models = envs_for_models
							)


if __name__ == '__main__':
	main()
