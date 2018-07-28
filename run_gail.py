import os
import argparse
import copy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from network_models.policy_dcgan import Policy_dcgan
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
from utils import generator


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to data', default='../../dataset/mnist_test_seq.npy')
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    # prepare log dir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # moving mnist 読み込み
    data = np.load(args.data_path)
    obs_shape = [3, 64, 64, 1]
    # generator
    gen = generator(data, args.batch_size)

    # policy net
    Policy = Policy_dcgan('policy', obs_shape=obs_shape, decode=True)
    Old_Policy = Policy_dcgan('old_policy', obs_shape=obs_shape, decode=True)

    # ppo学習インスタンス
    PPO = PPOTrain(Policy, Old_Policy, obs_shape=obs_shape, gamma=args.gamma)
    # discriminator
    D = Discriminator(obs_shape=obs_shape)

    # tensorflow saver
    saver = tf.train.Saver()
    # session config
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))
    # start session
    with tf.Session(config=config) as sess:
        # summary writer
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # initialize Session
        sess.run(tf.global_variables_initializer())
        # episode loop
        for iteration in tqdm(range(args.iteration)):
            # create batch
            expert_batch = next(gen)
            # first 3 frame
            agent_batch = expert_batch[:,:3,:,:,:]

            # test
            #print('act_prob_op:', Policy.test_run(next(gen)[:,:3,:,:,:]))

            # buffer
            observations = []
            next_observations = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            # run episode
            while True:
                # inference action(next frame) and value
                act, v_pred = Policy.act(obs=agent_batch)

                # create next_obs
                agent_batch_next = np.concatenate([agent_batch[:,1:,:,:,:], act], axis=1)

                # inference reward by discriminator
                reward = D.get_rewards(agent_s=agent_batch, agent_a=agent_batch_next)

                # episodeの各変数を追加
                observations.append(agent_batch)
                next_observations.append(agent_batch_next)
                v_preds.append(v_pred)
                rewards.append(reward)

                run_policy_steps += 1
                if run_policy_steps >= 7:
                    v_preds_next = v_preds[1:] + [np.zeros(v_preds[0].shape)]
                    break
                else:
                    # updata observations by old observations
                    agent_batch = agent_batch_next

            writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(
                        tag='episode_reward',
                        simple_value=sum(rewards[-1]))]),
                    iteration)

            # discriminator
            D_step = 2
            D_expert = np.random.randint(low=0, high=5, size=D_step)
            D_agent = np.random.randint(low=0, high=len(observations), size=D_step)
            for i in range(D_step):
                # expert input
                expert_obs = expert_batch[:,D_expert[i]:D_expert[i]+3,:,:,:]
                expert_obs_next = expert_batch[:,D_expert[i]+1:D_expert[i]+4,:,:,:]
                # agent input
                agent_obs = observations[D_agent[i]]
                agent_obs_next = next_observations[D_agent[i]]
                # run discriminator train
                D.train(expert_s=expert_obs,
                        expert_a=expert_obs_next,
                        agent_s=agent_obs,
                        agent_a=agent_obs_next)

            # updata policy using PPO
            # get d_rewards from discrminator
            d_rewards = []
            for i, _ in enumerate(observations):
                d_reward = D.get_rewards(agent_s=observations[i], agent_a=next_observations[i])
                # transform d_rewards to numpy for placeholder
                d_rewards.append(d_reward)

            # get generalized advantage estimator
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            # gae = (gaes - gaes.mean()) / gaes.std()

            # assign parameters to old policy
            PPO.assign_policy_parameters()

            # sample index
            PPO_step = 6
            sample_indices = np.random.randint(
                    low=0,
                    high=len(observations),
                    size=PPO_step)
            # train PPO
            for epoch in range(PPO_step):
                idx = sample_indices[epoch]
                # run ppo
                PPO.train(
                        obs=observations[idx],
                        actions=next_observations[idx],
                        gaes=gaes[idx],
                        rewards=d_rewards[idx],
                        v_preds_next=v_preds_next[idx])

            # get summary
            summary = PPO.get_summary(
                    obs=observations[-1],
                    actions=next_observations[-1],
                    gaes=gaes[-1],
                    rewards=d_rewards[-1],
                    v_preds_next=v_preds_next[-1])

            # add summary
            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
