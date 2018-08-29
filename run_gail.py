import os
import copy
import json
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from network_models.dcgan_policy import DCGANPolicy
from network_models.sngan_policy import SNGANPolicy
from network_models.dcgan_discriminator import DCGANDiscriminator
from network_models.sngan_discriminator import SNGANDiscriminator
from algo.ppo import PPOTrain
from algo.trpo import TRPOTrain
from utils import generator


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data',
            default='../../dataset/mnist_test_seq.npy')
    parser.add_argument('--outdir', type=str, help='output directory',
            default='out')
    parser.add_argument('--algo', type=str, help='algo name, ppo or trpo',
            default='ppo')
    parser.add_argument('--iteration', type=int, help='iteration',
            default=int(1e3))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--obs_size', type=int, default=64)
    parser.add_argument('--D_step', type=int, default=2)
    parser.add_argument('--G_step', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--initial_lr', type=float, help='initial learningrate',
            default=1e-4)
    parser.add_argument('--lr_schedules', type=int, help='learningrate schedules',
            default=100)
    parser.add_argument('--c_vf', type=float, default=0.2)
    parser.add_argument('--vf_clip', type=str, default='')
    parser.add_argument('--c_entropy', type=float, default=0.01)
    parser.add_argument('--c_l1', type=float, default=1.0)
    parser.add_argument('--leaky', type=bool, default=True)
    parser.add_argument('--g_sn', type=bool, default='')
    parser.add_argument('--d_sn', type=bool, default='')
    parser.add_argument('--frequency', type=int, default=50)
    parser.add_argument('--gpu_num', type=str, help='specify GPU number',
            default='0')
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Prepare output directory
    # output directory counter
    out_counter = len([None for out in os.listdir(args.outdir) if args.algo in out])
    out = os.path.join(args.outdir, args.algo + '_' + str(out_counter + 1))
    if not os.path.exists(out):
        os.makedirs(out)

    config = {'data': args.data_path,
            'algo': args.algo,
            'batch_size': args.batch_size,
            'iteration': args.iteration,
            'D_step': args.D_step,
            'G_step': args.G_step,
            'gamma': args.gamma,
            'initia_learning_rate': args.initial_lr,
            'lr_schedules': args.lr_schedules,
            'c_vf': args.c_vf,
            'c_entropy': args.c_entropy,
            'c_l1': args.c_l1,
            'vf_clip': args.vf_clip,
            'leaky': args.leaky,
            'SNGANGenerator': args.g_sn,
            'SNGANDiscriminator': args.d_sn}
    with open(os.path.join(out, 'config.json'), 'w') as f:
        f.write(json.dumps(config, indent=4))

    # Load moving mnist
    data = np.load(args.data_path)
    obs_shape = [3, args.obs_size, args.obs_size, 1]
    # Start generator
    gen = generator(data,
            batch_size=args.batch_size, img_size=args.obs_size)

    if not args.g_sn:
        print('Building DCGAN Generator as Policy Network')
        # Build policy network
        Policy = DCGANPolicy(
                'policy',
                obs_shape=obs_shape,
                batch_size=args.batch_size,
                decode=True,
                leaky=args.leaky)
        Old_Policy = DCGANPolicy(
                'old_policy',
                obs_shape=obs_shape,
                batch_size=args.batch_size,
                decode=True,
                leaky=args.leaky)
    else:
        print('Building SNGAN Generator as Policy Network')
        Policy = SNGANPolicy(
                'policy',
                obs_shape=obs_shape,
                batch_size=args.batch_size,
                decode=True)
        Old_Policy = SNGANPolicy(
                'old_policy',
                obs_shape=obs_shape,
                batch_size=args.batch_size,
                decode=True)

    # Build reinforcement agent
    if args.algo == 'ppo':
        print('Building PPO Agent')
        Agent = PPOTrain(
                Policy,
                Old_Policy,
                obs_shape=obs_shape,
                gamma=args.gamma,
                c_vf=args.c_vf,
                c_entropy=args.c_entropy,
                c_l1=args.c_l1,
                obs_size=args.obs_size,
                vf_clip=args.vf_clip)
    elif args.algo == 'trpo':
        print('Building TRPO Agent')
        Agent = TRPOTrain(
                Policy,
                Old_Policy,
                obs_shape=obs_shape,
                gamma=args.gamma,
                c_vf=args.c_vf,
                c_entropy=args.c_entropy,
                c_l1=args.c_l1,
                obs_size=args.obs_size,
                vf_clip=args.vf_clip)
    else:
        raise ValueError('invalid algo name')

    # Build Discriminator
    if args.d_sn:
        print('Building SNGAN Discriminator')
        D = SNGANDiscriminator(obs_shape=obs_shape,
            batch_size=args.batch_size)
    else:
        print('Building DCGAN Discriminator')
        D = DCGANDiscriminator(obs_shape=obs_shape,
            batch_size=args.batch_size,
            leaky=args.leaky)

    # Create tensorflow saver
    saver = tf.train.Saver()
    # session config
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))
    # Start session
    with tf.Session(config=config) as sess:
        # Summary writer
        writer = tf.summary.FileWriter(out, sess.graph)
        # Initialize Session
        sess.run(tf.global_variables_initializer())

        # Initialize learningrate
        lr = args.initial_lr
        # Start episode roop
        for iteration in tqdm(range(1, args.iteration+1)):
            if iteration % args.lr_schedules == 0:
                lr = lr / 10
                print('Decay learning rate')
            # Greate batch 0~1
            expert_batch = next(gen)
            # first 3 frame
            agent_batch = expert_batch[:, :3, :, :, :]

            # test
            #print('test: ', Policy.test_run(agent_batch))

            # buffer
            observations = []
            next_observations = []
            expert_actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            # Run episode
            while True:
                # Add episode
                observations.append(agent_batch)

                # Inference action(next frame) and value
                act, v_pred = Policy.act(obs=agent_batch)
                v_preds.append(v_pred)

                # Get expert actions
                expert_act = expert_batch[:, run_policy_steps+3, :, :, :]
                expert_act = np.reshape(expert_act,
                        (args.batch_size, args.obs_size * args.obs_size))
                expert_actions.append(expert_act)

                # Create next_obs
                act = np.reshape(act,
                        (args.batch_size, 1, args.obs_size, args.obs_size, 1))
                agent_batch_next = np.concatenate(
                        [agent_batch[:, 1:3, :, :, :], act],
                        axis=1)
                next_observations.append(agent_batch_next)

                run_policy_steps += 1
                if run_policy_steps >= 7:
                    v_preds_next = v_preds[1:] + [np.zeros(v_preds[0].shape)]
                    break
                else:
                    # Updata observations by old observations
                    agent_batch = agent_batch_next

            # Discriminator
            D_step = args.D_step
            for i, _ in enumerate(observations):
                # Get observations of expert and policy
                expert_obs = expert_batch[:, i:i+3, :, :, :]
                expert_obs_next = expert_batch[:, i+1:i+4, :, :, :]
                agent_obs = observations[i]
                agent_obs_next = next_observations[i]
                # Train discriminator
                _, D_loss = D.train(expert_s=expert_obs,
                        expert_s_next=expert_obs_next,
                        agent_s=agent_obs,
                        agent_s_next=agent_obs_next,
                        lr=args.initial_lr)
            print('D_loss: ', D_loss)

            # Updata policy using Agent
            # Get d_rewards from discrminator
            d_rewards = []
            for i, _ in enumerate(observations):
                d_reward = D.get_rewards(agent_s=observations[i],
                        agent_s_next=next_observations[i])
                d_rewards.append(d_reward)
            episode_d_rewards = [d_reward.sum() for d_reward in d_rewards]
            print('D_rewards: average {}, first {}, last {}'.format(
                sum(episode_d_rewards) / len(observations),
                d_rewards[0].sum(), d_rewards[-1].sum()))

            writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(
                        tag='episode_reward',
                        simple_value=sum(episode_d_rewards))]),
                    iteration)

            # Get generalized advantage estimator
            gaes = Agent.get_gaes(rewards=d_rewards, v_preds=v_preds,
                    v_preds_next=v_preds_next)
            #gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)

            # Train Agent
            Agent_step = args.G_step
            # Assign parameters to old policy
            Agent.assign_policy_parameters()
            total_loss = 0
            clip_loss = 0
            vf_loss = 0
            entropy_loss = 0
            l1_loss = 0
            for i, _ in enumerate(observations):
                # Run ppo train operation
                losses = Agent.train(
                        obs=observations[i],
                        gaes=gaes[i],
                        rewards=d_rewards[i],
                        v_preds_next=v_preds_next[i],
                        expert_act=expert_actions[i],
                        lr=lr)
                total_loss += losses[1]
                clip_loss += losses[2]
                vf_loss += losses[3]
                entropy_loss += losses[4]
                l1_loss += losses[5]

            total_loss /= len(observations)
            clip_loss /= len(observations)
            vf_loss /= len(observations)
            entropy_loss /= len(observations)
            l1_loss /= len(observations)
            print('total_loss: {}, clip_loss: {}, vf_loss: {}, entropy_loss: {} l1_loss: {}'.format(
                total_loss, clip_loss, vf_loss, entropy_loss, l1_loss))

            # Get agent summary
            Agent_summary = Agent.get_summary(
                    obs=observations[-1],
                    gaes=gaes[-1],
                    rewards=d_rewards[-1],
                    v_preds_next=v_preds_next[-1],
                    expert_act=expert_actions[-1],
                    lr=lr)
            writer.add_summary(Agent_summary, iteration)

            # Save trained model
            if iteration % args.frequency == 0:
                saver.save(sess, out+'/model-{}.ckpt'.format(iteration))
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
