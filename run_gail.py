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
    parser.add_argument('--logdir', help='log directory', default='log')
    parser.add_argument('--savedir', help='save directory', default='trained_models')
    parser.add_argument('--algo', default='gail')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--D_step', default=2, type=int)
    parser.add_argument('--G_step', default=6, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--c_vf', default=0.2, type=float)
    parser.add_argument('--c_entropy', default=0.01, type=float)
    parser.add_argument('--c_l1', default=1.0, type=float)
    parser.add_argument('--leaky', default=True)
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    # create trained model dir
    # ckpt counter
    ckpt_counter = len([ckpt_dir for ckpt_dir in os.listdir(args.savedir) if args.algo in ckpt_dir])
    model_dir = os.path.join(args.savedir, args.algo + '_' + str(ckpt_counter + 1))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create log dir
    # log counter
    log_counter = len([log_dir for log_dir in os.listdir(args.logdir) if args.algo in log_dir])
    log_dir = os.path.join(args.logdir, args.algo + '_' + str(log_counter + 1))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # moving mnist 読み込み
    data = np.load(args.data_path)
    obs_shape = [3, 64, 64, 1]
    # generator
    gen = generator(data, args.batch_size)

    # policy net
    Policy = Policy_dcgan(
            'policy',
            obs_shape=obs_shape,
            decode=True,
            leaky=args.leaky)
    Old_Policy = Policy_dcgan(
            'old_policy',
            obs_shape=obs_shape,
            decode=True,
            leaky=args.leaky)
    # ppo学習インスタンス
    PPO = PPOTrain(
            Policy,
            Old_Policy,
            obs_shape=obs_shape,
            gamma=args.gamma,
            lr=args.learning_rate,
            c_vf=args.c_vf,
            c_entropy=args.c_entropy,
            c_l1=args.c_l1)

    # discriminator
    D = Discriminator(
            obs_shape=obs_shape,
            lr=args.learning_rate,
            leaky=args.leaky)

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
            #print('test: ', Policy.test_run(agent_batch))

            # buffer
            observations = []
            next_observations = []
            expert_actions = []
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

                # get expert actions
                expert_act = expert_batch[:, run_policy_steps+3, :, :, :]
                expert_act = np.reshape(expert_act, (args.batch_size, 64 * 64))
                expert_actions.append(expert_act)

                # create next_obs
                act = np.reshape(act, (args.batch_size, 1, 64, 64, 1))
                agent_batch_next = np.concatenate([agent_batch[:, 1:3, :, :, :], act], axis=1)
                next_observations.append(agent_batch_next)

                run_policy_steps += 1
                if run_policy_steps >= 7:
                    v_preds_next = v_preds[1:] + [np.zeros(v_preds[0].shape)]
                    break
                else:
                    # updata observations by old observations
                    agent_batch = agent_batch_next

            # discriminator
            D_step = args.D_step
            # 動画の全タイムステップで学習
            for i, _ in enumerate(observations):
                # get observations of expert and policy
                expert_obs = expert_batch[:, i:i+3, :, :, :]
                expert_obs_next = expert_batch[:, i+1:i+4, :, :, :]
                agent_obs = observations[i]
                agent_obs_next = next_observations[i]
                # train discriminator
                _, D_loss = D.train(expert_s=expert_obs,
                        expert_s_next=expert_obs_next,
                        agent_s=agent_obs,
                        agent_s_next=agent_obs_next)

            print('D_loss: ', D_loss)

            '''
            # get Discriminator summary
            D_summary = D.get_summary(
                    expert_s=expert_batch[:,6:9,:,:,:],
                    expert_s_next=expert_batch[:,7:10,:,:,:],
                    agent_s=observations[-1],
                    agent_s_next=next_observations[-1]
                    )
            # add Discriminator summary
            writer.add_summary(D_summary, iteration)
            '''

            # updata policy using PPO
            # get d_rewards from discrminator
            d_rewards = []
            for i, _ in enumerate(observations):
                d_reward = D.get_rewards(agent_s=observations[i], agent_s_next=next_observations[i])
                d_rewards.append(d_reward)
            print('D_rewards: first {}, last {}'.format(sum(d_rewards[0]), sum(d_rewards[-1])))

            writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(
                        tag='episode_reward',
                        simple_value=sum(d_rewards[0]))]),
                    iteration)

            # get generalized advantage estimator
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            #gaes = [r_t + PPO.gamma * v_next - v for r_t, v_next, v in zip(d_rewards, v_preds_next, v_preds)]
            # gae = (gaes - gaes.mean()) / gaes.std()


            # train PPO
            PPO_step = args.G_step
            # assign parameters to old policy
            PPO.assign_policy_parameters()
            for i, _ in enumerate(observations):
                # run ppo train operation
                _, total_loss, clip_loss, vf_loss, entropy_loss, l1_loss = PPO.train(
                        obs=observations[i],
                        gaes=gaes[i],
                        rewards=d_rewards[i],
                        v_preds_next=v_preds_next[i],
                        expert_act=expert_actions[i])
                '''
                gradients = PPO.get_grad(
                        obs=observations[i],
                        gaes=gaes[i],
                        rewards=d_rewards[i],
                        v_preds_next=v_preds_next[i])
                '''
            print('total_loss: {}, clip_loss: {}, vf_loss: {}, entropy_loss: {} l1_loss: {}'.format(
                total_loss, clip_loss, vf_loss, entropy_loss, l1_loss))

            # get PPO summary
            PPO_summary = PPO.get_summary(
                    obs=observations[-1],
                    gaes=gaes[-1],
                    rewards=d_rewards[-1],
                    v_preds_next=v_preds_next[-1],
                    expert_act=expert_actions[-1])
            # add PPO summary
            writer.add_summary(PPO_summary, iteration)

            # save trained model
            if iteration % 500 == 0:
                saver.save(sess, model_dir+'/model-{}.ckpt'.format(iteration))
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
