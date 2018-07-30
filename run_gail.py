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
    parser.add_argument('--logdir', help='log directory', default='log/train')
    parser.add_argument('--savedir', help='save directory', default='trained_models')
    parser.add_argument('--algo', default='gail')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--learning_rate', default=1e-2)
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # ckpt counter
    ckpt_counter = len([ckpt_dir for ckpt_dir in os.listdir(args.savedir) if args.algo in ckpt_dir])
    model_dir = args.savedir + '/' + args.algo + '_' + str(ckpt_counter + 1)
    # create trained model dir
    if not os.path.exists(args.savedir):
        os.makedirs(model_dir)

    # log counter
    log_counter = len([log_dir for log_dir in os.listdir(args.logdir) if args.algo in log_dir])
    log_dir = args.logdir + '/' + args.algo + '_' + str(log_counter + 1)
    # create log dir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # moving mnist 読み込み
    data = np.load(args.data_path)
    obs_shape = [3, 64, 64, 1]
    # generator
    gen = generator(data, args.batch_size)

    # policy net
    Policy = Policy_dcgan(
            'policy',
            obs_shape=obs_shape,
            decode=True)
    Old_Policy = Policy_dcgan(
            'old_policy',
            obs_shape=obs_shape,
            decode=True)

    # ppo学習インスタンス
    PPO = PPOTrain(
            Policy,
            Old_Policy,
            obs_shape=obs_shape,
            gamma=args.gamma,
            lr=args.learning_rate)
    # discriminator
    D = Discriminator(obs_shape=obs_shape, lr=args.learning_rate)

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
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # initialize Session
        sess.run(tf.global_variables_initializer())
        # episode loop
        for iteration in tqdm(range(1, args.iteration+1)):
            # create batch 0~1
            expert_batch = next(gen)
            expert_batch = expert_batch / 255
            # first 3 frame
            agent_batch = expert_batch[:,:3,:,:,:]

            # test
            print('test: ', Policy.test_run(agent_batch))

            # buffer
            observations = []
            next_observations = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            # run episode
            while True:
                # episodeの各変数を追加
                observations.append(agent_batch)

                # inference action(next frame) and value
                act, v_pred = Policy.act(obs=agent_batch)
                v_preds.append(v_pred)

                # create next_obs
                act = np.reshape(act, (args.batch_size,1,64,64,1))
                agent_batch_next = np.concatenate([agent_batch[:,1:,:,:,:], act], axis=1)
                next_observations.append(agent_batch_next)


                # inference reward by discriminator
                reward = D.get_rewards(agent_s=agent_batch, agent_s_next=agent_batch_next)
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
                        simple_value=sum(rewards[0]))]),
                    iteration)

            # discriminator
            D_step = 2
            D_indices = np.random.randint(
                    low=0,
                    high=6,
                    size=D_step)
            for i, idx in enumerate(D_indices):
                # expert input
                expert_obs = expert_batch[:, idx:idx+3, :, :, :]
                expert_obs_next = expert_batch[:, idx+1:idx+4, :, :, :]
                # agent input
                agent_obs = observations[idx]
                agent_obs_next = next_observations[idx]
                # run discriminator train operation
                D.train(expert_s=expert_obs,
                        expert_s_next=expert_obs_next,
                        agent_s=agent_obs,
                        agent_s_next=agent_obs_next)
            '''
            # get Discriminator summary
            D_summary = D.get_summary(
                    expert_s=expert_batch[:,6:9,:,:,:],
                    expert_s_next=expert_batch[:,7:10,:,:,:],
                    agent_s=agent_batch[-1],
                    agent_s_next=agent_batch_next[-1])
            # add Discriminator summary
            writer.add_summary(D_summary, iteration)
            '''

            # updata policy using PPO
            # get d_rewards from discrminator
            d_rewards = []
            for i, _ in enumerate(observations):
                d_reward = D.get_rewards(agent_s=observations[i], agent_s_next=next_observations[i])
                # transform d_rewards to numpy for placeholder
                d_rewards.append(d_reward)

            # get generalized advantage estimator
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            # gae = (gaes - gaes.mean()) / gaes.std()

            # assign parameters to old policy
            PPO.assign_policy_parameters()

            # sample index
            PPO_step = 2
            sample_indices = np.random.randint(
                    low=0,
                    high=len(observations),
                    size=PPO_step)

            # train PPO
            for i, idx in enumerate(sample_indices):
                # run ppo train operation
                PPO.train(
                        obs=observations[idx],
                        gaes=gaes[idx],
                        rewards=d_rewards[idx],
                        v_preds_next=v_preds_next[idx])

            # get PPO summary
            PPO_summary = PPO.get_summary(
                    obs=observations[-1],
                    gaes=gaes[-1],
                    rewards=d_rewards[-1],
                    v_preds_next=v_preds_next[-1])
            # add PPO summary
            writer.add_summary(PPO_summary, iteration)

            # save trained model
            if (iteration) % 1000 == 0:
                saver.save(sess, model_dir+'/model-{}.ckpt'.format(iteration))
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
