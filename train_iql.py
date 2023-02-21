import os
from typing import Tuple
from pathlib import Path
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb
import sys
import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from utils import get_user_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './result/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
# network architecture
flags.DEFINE_boolean('encoder', True, 'an encoder for actor and critic input')
flags.DEFINE_enum('rep_module', 'backbone', ['backbone', 'encoder'], 'The network for representation learning')
# load weight
flags.DEFINE_boolean('bc_eval', False, '')
flags.DEFINE_integer('weight_num', 3, 'how many weights to compute avg')
flags.DEFINE_string('weight_ensemble', 'mean', 'how to aggregate weights over runnings')
flags.DEFINE_string('weight_path', '', 'bc adv path str pattern')
flags.DEFINE_integer('iter', 1, 'K th rebalanced behavior policy used for offrl training.')
flags.DEFINE_enum('weight_func', 'linear', ['linear', 'exp', 'power'], '')
flags.DEFINE_float('std', 2.0, help="scale weights' standard deviation.")
flags.DEFINE_float('eps', 0.1, '')
flags.DEFINE_float('eps_max', None, '')
flags.DEFINE_float('exp_lambd', 1.0, '')
flags.DEFINE_boolean('pb', False, 'progressive-balanced sampling')
flags.DEFINE_integer('pb_step', int(8e5), '')
flags.DEFINE_integer('pb_interval', 1000, 'Eval interval.')
# train
flags.DEFINE_integer('train_steps', int(1e6), '')
flags.DEFINE_enum('sampler', 'uniform', ['uniform', 'return-balance', 'inverse-return-balance'], '')
flags.DEFINE_boolean('two_sampler', False, '')
flags.DEFINE_boolean('reinitialize', False, 'reinitialize the output layer')
flags.DEFINE_boolean('reweight', False, '')
flags.DEFINE_boolean('reweight_eval', True, '')
flags.DEFINE_boolean('reweight_improve', True, '')
flags.DEFINE_boolean('reweight_constraint', True, '')
flags.DEFINE_boolean('grad_clip', False, '')
flags.DEFINE_float('max_grad_norm', 10.0, '')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_string('tag', '', 'tag of the run.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    dataset = D4RLDataset(env, FLAGS.batch_size, FLAGS.sampler, FLAGS.reweight, FLAGS.config.base_prob, pb=FLAGS.pb)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    assert FLAGS.sampler == 'uniform' or not FLAGS.reweight 
    assert not (FLAGS.reweight_improve ^ FLAGS.reweight_constraint)
    kwFLAGS = dict(FLAGS.config)
    # set up wandb
    wandb.init(project="IQL-reweight-v2", config={"env": FLAGS.env_name, "seed": FLAGS.seed,
            "encoder": FLAGS.encoder,  "sampler": FLAGS.sampler, "two_sampler": FLAGS.two_sampler, "train_steps": FLAGS.train_steps, 
            "encoder_hidden_dims": FLAGS.config.encoder_hidden_dims, "embedding_dim": FLAGS.config.embedding_dim, "hidden_dims": FLAGS.config.hidden_dims,
            "base_prob": FLAGS.config.base_prob, "expectile": FLAGS.config.expectile, "temperature": FLAGS.config.temperature,
            "reweight": FLAGS.reweight, "reweight_eval": FLAGS.reweight_eval,
            "reweight_improve": FLAGS.reweight_improve, "reweight_constraint": FLAGS.reweight_constraint,
            "grad_clip": FLAGS.grad_clip, "max_grad_norm": FLAGS.max_grad_norm,
            # load weights
            "bc_eval": FLAGS.bc_eval, "iter": FLAGS.iter,  "weight_func": FLAGS.weight_func,
            "std": FLAGS.std, "eps": FLAGS.eps, "eps_max": FLAGS.eps_max, "weight_ensemble": FLAGS.weight_ensemble,
            "tag": FLAGS.tag})

    FLAGS.save_dir = Path(os.path.join(FLAGS.save_dir, FLAGS.tag, FLAGS.env_name, str(FLAGS.seed)))
    FLAGS.save_dir = Path(os.path.join(FLAGS.save_dir, FLAGS.tag, FLAGS.env_name, str(FLAGS.seed)))
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb'),
                                #    write_to_disk=True)
                                   write_to_disk=False)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    if FLAGS.bc_eval:
        # weight loading module (filename changed)
        weight_list = []
        for seed in range(1, FLAGS.weight_num + 1):
            try:
                wp = FLAGS.weight_path%seed
            except:
                wp = FLAGS.weight_path # load the speificed weight
            eval_res = np.load(wp, allow_pickle=True).item()
            try:
                num_iter, bc_eval_steps = eval_res['iter'], eval_res['eval_steps']
                assert FLAGS.iter <= num_iter
                t = eval_res[FLAGS.iter]
            except:
                assert FLAGS.iter == 1
                t = eval_res[1000000]['adv']
            weight_list.append(t)
            print(f'Loading weights from {wp} at {FLAGS.iter}th rebalanced behavior policy')
        if FLAGS.weight_ensemble == 'mean':
          weight = np.stack(weight_list, axis=0).mean(axis=0)
        elif FLAGS.weight_ensemble == 'median':
          weight = np.median(np.stack(weight_list, axis=0), axis=0)
        else:
          raise NotImplementedError
        assert dataset.size == weight.shape[0]
        dataset.replace_weights(weight, FLAGS.weight_func, FLAGS.exp_lambd, FLAGS.std, FLAGS.eps, FLAGS.eps_max)


    if FLAGS.train_steps > 0:

        eval_returns = []
        rep_agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.train_steps,
                    finetune=None,
                    encoder = FLAGS.encoder,
                    reweight_eval = FLAGS.reweight_eval,
                    reweight_improve = FLAGS.reweight_improve,
                    reweight_constraint = FLAGS.reweight_constraint,
                    grad_clip=FLAGS.grad_clip,
                    max_gradient_norm=FLAGS.max_grad_norm,
                    **kwFLAGS)
        for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            batch = dataset.sample()
            if FLAGS.two_sampler:
                uni_batch = dataset.sample(uniform=True)
            else:
                uni_batch = batch

            update_info = rep_agent.update(batch, uni_batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                        wandb.log({f"training_{k}": v}, step=i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                eval_episode = max(100, FLAGS.eval_episodes) if i ==  FLAGS.train_steps else FLAGS.eval_episodes
                eval_stats = evaluate(rep_agent, env, eval_episode)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
                    wandb.log({f'evaluation/{k}': v}, step=i)

                summary_writer.flush()
            
            if FLAGS.pb and i % FLAGS.pb_interval == 0:
                T = int(FLAGS.pb_step / FLAGS.pb_interval)
                t = int(i / FLAGS.pb_interval)
                max_weight = dataset.progressive_balance(t, T)
                wandb.log({f"training_max_weight": max_weight}, step=i)
                # eval_returns.append((i, eval_stats['return']))
                # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                #            eval_returns,
                #            fmt=['%d', '%.1f'])

        # save and load
        # rep_agent.save(FLAGS.save_dir / 'ckpt')
        # os._exit(os.EX_OK)



if __name__ == '__main__':
    app.run(main)
