import hydra
import os
import torch
from replay_buffer import PrefetchBalancedSampler, sample, split_traj_and_compute_return, get_topn_replay
import numpy as np
from copy import deepcopy
from experiment_logging import default_logger as logger
import wandb
PRETRAIN_NUM = 0  # the number of trained beta and Q before running
@hydra.main(config_path='config', config_name='train')
def train(cfg):
    print('jobname: ', cfg.name)

    wandb.init(project="onestep", config={'env': cfg.env.name, 'seed': cfg.seed, 
                        'temp': cfg.pi.temp, 'resample': cfg.resampling, 'topn': cfg.topn, 'std': cfg.std,
                        'iter': cfg.iter, 'eps': cfg.eps, 'eps_max': cfg.eps_max, 'temp': cfg.pi.temp,
                        'model_tag': cfg.model_tag, 'tag': cfg.tag,})

    # load data
    replay = torch.load(cfg.data_path)


    # load env
    import gym
    import d4rl
    if cfg.env.name[-2:] != 'v2' and cfg.env.name[-2:] != 'v1' \
        and cfg.env.name[-2:] != 'v0': 
        name_list = cfg.env.name.split('v2')
        env = gym.make(name_list[0] + 'v2')
    else:
        env = gym.make(cfg.env.name)
    cfg.state_dim = int(np.prod(env.observation_space.shape))
    cfg.action_dim = int(np.prod(env.action_space.shape))

    # set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # build learners
    q = hydra.utils.instantiate(cfg.q)
    pi = hydra.utils.instantiate(cfg.pi)
    beta = hydra.utils.instantiate(cfg.beta)
    baseline = hydra.utils.instantiate(cfg.baseline)

    if cfg.resampling == 'adv':
        weight_list = []
        for seed in range(1, cfg.weight_num + 1):
            try:
                wp = cfg.weight_path%seed
            except:
                wp = cfg.weight_path # load the speificed weight
            eval_res = np.load(wp, allow_pickle=True).item()
            try:
                num_iter, bc_eval_steps = eval_res['iter'], eval_res['eval_steps']
                assert cfg.iter <= num_iter
                t = eval_res[cfg.iter]
            except:
                assert cfg.iter == 1
                t = eval_res[1000000]['adv']
            weight_list.append(t)
            print(f'Loading weights from {wp} at {cfg.iter}th rebalanced behavior policy')
        if cfg.weight_ensemble == 'mean':
          weight = np.stack(weight_list, axis=0).mean(axis=0)
        elif cfg.weight_ensemble == 'median':
          weight = np.median(np.stack(weight_list, axis=0), axis=0)
        else:
          raise NotImplementedError
        assert replay.length == weight.shape[0]
        # TODO: scale
        if cfg.weight_func == 'linear':
            weight = weight - weight.min()
            prob = weight / weight.sum()
            # keep mean, scale std
            if cfg.std:
                scale = cfg.std / (prob.std() * replay.length)
                prob = scale*(prob - 1/replay.length) + 1/replay.length
                if cfg.eps: # if scale, the prob may be negative.
                    prob = np.maximum(prob, cfg.eps/replay.length)
                if cfg.eps_max: # if scale, the prob may be too large.
                    prob = np.minimum(prob, cfg.eps_max/replay.length)
            prob = prob/prob.sum() # norm to 1 again
        assert q.batch_size == pi.batch_size == beta.batch_size
        sampler = PrefetchBalancedSampler(prob, replay.length, q.batch_size, 1000)
        replay.sampler = sampler
        # set new resampling method
        import types
        replay.sample = types.MethodType(sample, replay)
    elif cfg.resampling == 'uniform' and cfg.topn < 100:
        split_traj_and_compute_return(replay)
        replay = get_topn_replay(replay, cfg.topn)
    elif cfg.resampling == 'return':
        split_traj_and_compute_return(replay)
        dist = replay.returns
        base_prob = 0.2 if 'antmaze' in cfg.env.name  else 0 
        prob = (dist - dist.min()) / (dist.max() - dist.min()) + base_prob
        sampler = PrefetchBalancedSampler(prob, replay.length, q.batch_size, 1000)
        replay.sampler = sampler
        import types
        replay.sample = types.MethodType(sample, replay)

    # setup logger 
    os.makedirs(cfg.log_dir, exist_ok=True)
    model_dir = os.path.split(cfg.beta.model_save_path)[0]
    os.makedirs(model_dir, exist_ok=True)
    setup_logger(cfg)
    q.set_logger(logger)
    pi.set_logger(logger)
    beta.set_logger(logger)
    baseline.set_logger(logger)

    if PRETRAIN_NUM > 0:
        model_ind = cfg.seed % PRETRAIN_NUM if cfg.seed % PRETRAIN_NUM != 0 else PRETRAIN_NUM
    else:
        model_ind = cfg.seed
    model_ind = int(model_ind)
    model_tag = cfg.model_tag or cfg.tag

    # train
    if cfg.pi.name == 'pi_easy_bcq':
        pi.update_beta(beta)
        pi.update_q(q)

    # train beta
    if cfg.train_beta:
        beta_model_path = os.path.join(cfg.path, 'models', model_tag, f'train_{cfg.env.name}_{model_ind}_{cfg.beta.name}') + '_' + str(int(cfg.beta_steps)) + f'_{model_ind}.pt'
        if os.path.exists(beta_model_path):
            beta.load(beta_model_path)
            print(f'load beta model from {beta_model_path}')
        else:
            for step in range(int(cfg.beta_steps)):
                beta.train_step(replay, None, None, None)
                if (step+1) % int(cfg.log_freq) == 0:
                    logger.update('beta/step', step)
                    logger.write_sub_meter('beta')
                if (step+1) % int(cfg.eval_freq) == 0:
                    ret, norm_ret = beta.eval(env, cfg.eval_episodes)
                    wandb.log({'beta/score': norm_ret}, step=step+1)
            beta_save_path = cfg.beta.model_save_path + '_' + str(int(cfg.beta_steps)) + f'_{cfg.seed}.pt'
            beta.save(beta_save_path)

    # train baseline
    if cfg.train_baseline:
        for step in range(int(cfg.baseline_steps)):
            baseline.train_step(replay)

            if (step+1) % int(cfg.log_freq) == 0:
                logger.update('baseline/step', step)
                logger.write_sub_meter('baseline')
            # if (step+1) % int(cfg.eval_freq) == 0:
            #     ret, norm_ret = baseline.eval(env, beta, cfg.eval_episodes)
            #     wandb.log({'baseline/score': norm_ret}, step=step)
        beta.save(cfg.beta.model_save_path + '_' + str(step+1) + f'_{cfg.seed}.pt')

    # load beta as init pi
    pi.load_from_pilearner(beta)

    # iterate between eval and improvement
    for out_step in range(int(cfg.steps)):        
        # train Q
        if cfg.train_q:
            q_model_path = os.path.join(cfg.path, 'models', model_tag, f'train_{cfg.env.name}_{model_ind}_q') + '_' + str(int(cfg.q_steps)) + f'_{model_ind}.pt'
            if os.path.exists(q_model_path):
                q.load(q_model_path)
                print(f'load q model from {q_model_path}')
            else:
                for in_step in range(int(cfg.q_steps)): 
                    q.train_step(replay, pi, beta)
                    
                    step = out_step * int(cfg.q_steps) + in_step 
                    if (step+1) % int(cfg.log_freq) == 0:
                        logger.update('q/step', step)
                        q.eval(env, pi, cfg.eval_episodes)
                        logger.write_sub_meter('q')
                q_save_path = cfg.q.model_save_path + '_' + str(int(cfg.q_steps)) + f'_{cfg.seed}.pt'
                q.save(q_save_path)

        # train pi
        if cfg.train_pi and cfg.pi.name != 'pi_easy_bcq':
            for in_step in range(int(cfg.pi_steps)):
                pi.train_step(replay, q, baseline, beta)

                step = out_step * int(cfg.pi_steps) + in_step
                if (step+1) % int(cfg.log_freq) == 0:
                    logger.update('pi/step', step)
                    logger.write_sub_meter('pi')
                if (step+1) % int(cfg.eval_freq) == 0:
                    ret, norm_ret = pi.eval(env, cfg.eval_episodes)
                    wandb.log({'pi/score': norm_ret}, step=int(cfg.beta_steps)+step+1)
            # pi.save(cfg.pi.model_save_path + '_' + str(step+1) + f'_{cfg.seed}.pt')
        elif cfg.pi.name == 'pi_easy_bcq':
            step = out_step + 1
            pi.update_q(q)
            if step % int(cfg.log_freq) == 0:
                logger.update('pi/step', step)
                pi.eval(env, cfg.eval_episodes)
                logger.write_sub_meter('pi')
    
    # if cfg.train_q:
    #     q.save(cfg.q.model_save_path + '.pt')
    # if cfg.train_pi:
    #     pi.save(cfg.pi.model_save_path + '.pt')
    # wandb.finish()
    exit()
     
def setup_logger(cfg):
    logger_dict = dict()
    if cfg.train_q:
        q_train_dict = {'q': {
                        'csv_path': f'{cfg.log_dir}/q.csv',
                        'format_str': cfg.q.format_str,
                    },} 
        logger_dict.update(q_train_dict)
    if cfg.train_pi or cfg.pi.name == 'pi_easy_bcq':
        pi_train_dict = {'pi': {
                        'csv_path': f'{cfg.log_dir}/pi.csv',
                        'format_str': cfg.pi.format_str,
                    },} 
        logger_dict.update(pi_train_dict)
    if cfg.train_beta:
        beta_train_dict = {'beta': {
                        'csv_path': f'{cfg.log_dir}/beta.csv',
                        'format_str': cfg.beta.format_str,
                    },} 
        logger_dict.update(beta_train_dict)
    if cfg.train_baseline:
        beta_train_dict = {'baseline': {
                        'csv_path': f'{cfg.log_dir}/baseline.csv',
                        'format_str': cfg.baseline.format_str,
                    },} 
        logger_dict.update(beta_train_dict)

    logger.setup(logger_dict, summary_format_str=None) 


if __name__ == "__main__":
    train()