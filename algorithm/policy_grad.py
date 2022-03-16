from plan import *
from database_env import DataBaseEnv_QueryEncoding
from utils.utils import *
from utils.db_utils import *
from tree import *
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter
import re

import logging
LOG = logging.getLogger(__name__)

INF = 1e9


# def loss_func(pred, target):
#     # target = [b_indices, dim_1_indices, dim_2_indices],  q-values
#     # pred.shape = [Nf, max(ni), max(ni)
#     # print(pred[target[0]][:10])
#     # print(target[1][:10])
#     # preds = torch.clamp(pred[target[0]], max=float(np.log(3*10**5)))
#     preds = pred[target[0]]
#     return torch.nn.functional.mse_loss(preds, target[1])


class EnvBase(DataBaseEnv_QueryEncoding):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.cardinalities = env_config.get("cardinalities")

    def apply_plan_encoding(self, plan):
        g = plan.G
        for n in g:
            node_info = g.nodes[n]
            if not "feature" in node_info:
                if self.cardinalities:
                    vec = np.zeros(len(self.rels)+1)
                    if 'name' in node_info:
                        vec[-1] = self.cardinalities[plan.initial_query][node_info['name']]
                else:
                    vec = np.zeros(len(self.rels))
                vec[[self.rel_to_idx[t] for t in node_info['tables']]] = 1
                g.nodes[n]["feature"] = vec.reshape(1, -1)

    def get_obs(self, plan=None):
        p = plan if plan else self.plan
        self.apply_plan_encoding(p)
        return super().get_obs(p), to_forest(p)


class EnvPlan(EnvBase):
    def __init__(self, env_config, true_reward=True):
        super().__init__(env_config)
        self.true_reward = true_reward

    def get_state(self):
        return self.get_obs(self.plan)

    def step(self, action):
        self.plan.join(*self.plan.action_to_join(action))
        is_complete = self.plan.is_complete
        cost = None
        if is_complete and self.true_reward:
            cost = self.reward()
        return None, cost, is_complete, None


# class EnvPlanBeam(EnvBase):
#     def __init__(self, env_config, beam_width, true_reward=True):
#         super().__init__(env_config)
#         self.true_reward = true_reward
#         self.beam_width = beam_width
#
#     def reset(self, idx=None):
#         super().reset(idx)
#         self.plans = [self.plan]
#
#     def get_states(self):
#         obs = []
#         for plan in self.plans:
#             obs.append(self.get_obs(plan))
#         return obs
#
#     def step(self, actions):
#         plans = {}
#         for i, a in enumerate(actions):
#             p = deepcopy(self.plans[i])
#             p.join(*p.action_to_join(a), action=a)
#             plans[p] = None
#             if len(plans) == self.beam_width:
#                 break
#         self.plans = list(plans.keys())
#         is_complete = self.plans[0].is_complete
#         cost = None
#         self.plan = self.plans[0]
#         if self.true_reward:
#             cost = self.reward()
#         return None, cost, is_complete, None


def FC(d_in, d_out, fc_nlayers, drop):
    dims = torch.linspace(d_in, d_out, fc_nlayers+1, dtype=torch.long)
    layers = []
    for i in range(fc_nlayers-1):
        layers.extend([nn.Linear(int(dims[i]), int(dims[i+1])),
                       nn.Dropout(drop), nn.LayerNorm([int(dims[i+1])]), nn.ReLU()])
    layers.append(nn.Linear(int(dims[-2]), d_out))
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, d_emb, d_query, d_model, d_pairwise, nhead, ffdim, nlayers, fc_nlayers, drop, pretrained_path=False, fit_pretrained_layers=[], **kwargs):
        super().__init__()
        self.args = {k: v for k, v in locals().items() if k not in [
                                             'self', '__class__']}
        # Tree transformer
        self.enc = nn.Linear(d_emb, (d_model+1)//2)
        self.trans_enc = nn.TransformerEncoder(
            TreeTransformerEncoderLayer(d_model, nhead, ffdim, drop), nlayers)
        self.cls = nn.Parameter(torch.empty((1, 1, d_model)))
        torch.nn.init.xavier_uniform_(self.cls, gain=1.0)
        # Pairwise module to get values for each possible action
        # self.key = torch.nn.Linear(d_model, d_pairwise)
        # self.query = torch.nn.Linear(d_model, d_pairwise)
        # FC layers
        self.key = FC(d_model, d_pairwise, fc_nlayers, drop)
        self.query = FC(d_model, d_pairwise, fc_nlayers, drop)
        self.val = FC(d_model, 1, fc_nlayers, drop)
        # Query level
        d_q = d_model // 2
        self.qn = nn.Sequential(nn.Linear(d_query, d_q))
        self.pretrained_path = pretrained_path
        self.fit_pretrained_layers = fit_pretrained_layers

    def forward(self, inputs):
        q, t = inputs
        q = self.qn(q).unsqueeze(0)  # [1, (n1+n2+...), d_model // 2]
        x, indices, lens = t
        # [L, (n1+n2+...), d_model // 2]; ni = number of trees in i-th forest
        x = self.enc(x)
        # [L, (n1+n2+...), d_model]
        x = torch.cat((x, q.expand(x.shape[0], -1, -1)), -1)
        x = torch.cat((self.cls.expand(-1, x.shape[1], -1), x), 0)
        x, _ = self.trans_enc((x, indices))  # [1, (n1+n2+...), d_model], ...
        l = torch.split(x[0], lens)
        x = torch.nn.utils.rnn.pad_sequence(
            l, batch_first=True)  # [Nf, max(ni), d_model]
        V = self.val(torch.mean(x, 1))  # [Nf, 1]
        k, q = self.key(x), self.query(x)
        P = (torch.matmul(k, q.transpose(1, 2))
             / np.sqrt(k.shape[-1]))  # [Nf, max(ni), max(ni)]
        return P, V

    # def new(self):
    #     if self.pretrained_path:
    #         model = deepcopy(self)
    #         model.load_state_dict(torch.load(self.pretrained_path))
    #         return model
    #     else:
    #         return self.__class__(**self.args)


def collate(batch):
    if isinstance(batch[0][0], np.ndarray):
        q_enc, forests = zip(*batch)
        lens = [len(f) for f in forests]
        flatten_tc_batch = flatten_batch_TreeConv([t.to_torch().to(
            dtype=torch.float) for f in forests for t in f], batch_first=False)
        q_enc = torch.repeat_interleave(torch.tensor(
            q_enc, dtype=torch.float), torch.tensor(lens), dim=0)
        # [n_sum, d_qenc], ([L,n_sum,d_enc],[3L,n_sum,1],[batch_size])
        return q_enc, (*flatten_tc_batch, lens)
    # batch is a list of trajectories with different length
    # make it flat first
    batch = [x for traj in batch for x in traj]
    x, y = zip(*batch)
    actions, valid_mask, done_mask, rewards = zip(*y)
    valid_mask = stack_masks(list(valid_mask))
    size = valid_mask.shape[1]
    # flat_actions = [list(range(len(actions))),
    #                 [a[0] for a in actions],
    #                 [a[1] for a in actions]]
    flat_actions = torch.tensor([a[0]*size + a[1]
                                 for i, a in enumerate(actions)]).view(-1, 1)
    done_mask = torch.tensor(done_mask)
    rewards = torch.tensor(rewards, dtype=torch.float)
    return collate(x), (flat_actions, valid_mask, done_mask, rewards)


class TrajectoryStorage():
    def __init__(self):
        self.episodes = []

    def set_env(self, env):
        self.env = env

    def split_trajectory(self, plan, reward):
        traj = []
        for i, (node, action) in enumerate(plan._joins[::-1]):
            plan.disjoin(node)
            obs = self.env.get_obs(deepcopy(plan))
            traj.append(
                [obs, (action, get_mask(plan), i == 0, (i == 0)*reward)])
        return traj[::-1]

    def append(self, plan, final_reward):
        self.episodes.append(self.split_trajectory(
            deepcopy(plan), final_reward))

    def get_dataset(self, n=1000):
        """Get last n trajectories"""
        return self.episodes[-n:]


def ac_loss(pred, actions, valid_mask, done_mask, rewards, gamma):
    # [Nf, max(ni), max(ni)], [Nf, 1]
    logits, values = pred
    shape = logits.shape
    # compute qvalues based on rewards and predicted values of the next state
    qvalues = torch.where(done_mask,
                          rewards, gamma * values.detach().roll(-1, 0))
    advantage = (qvalues.view(-1) - values)
    masked_logits = torch.where(valid_mask,
                                logits, -torch.tensor(INF).to(valid_mask.device))
    # [Nf, max(ni), max(ni)]
    log_probs = (masked_logits
                 - masked_logits.logsumexp(dim=[1, 2], keepdim=True))
    log_probs = log_probs.view(shape[0], -1)  # [Nf, max(ni)*max(ni)]
    log_prob_action = torch.gather(log_probs, 1, actions).view(-1, 1)  # [Nf]
    probs = torch.exp(log_probs)  # [Nf, max(ni)*max(ni)]
    # mse
    value_loss = advantage.pow(2).mean()
    # policy grad loss
    policy_loss = -(log_prob_action*advantage.detach()).mean()
    # - entropy
    entropy_loss = torch.sum(log_probs*probs
                             / torch.sum(valid_mask, dim=[1, 2]).view(-1, 1) / shape[0])
    return policy_loss, value_loss, entropy_loss


# def ppo_loss(self, actions, returns, values, logits, old_probs, value_coeff=1., entropy_coef=1.):
#         advantage = returns - values # [length, batch_size if != 1]
#         # compute log(pi)
#         log_probs = torch.log_softmax(logits, dim=-1) # [length, batch_size, n_actions]
#         probs = torch.softmax(logits,dim=-1)
#         new_probs =  probs.gather(-1,actions).squeeze() # [length, batch_size if != 1]
#         # compute loss
#         value_loss = advantage.pow(2).mean()
#         # Clipped version of value loss
#         # value_obj = advantage.pow(2)
#         # value_obj2 = (returns - old_values + torch.clamp(values-old_values, -self.eps, self.eps)).pow(2)
#         # value_loss = torch.mean(torch.max(value_obj,value_obj2))
#         entropy = -(log_probs*probs).sum(-1).mean()
#         ratio = new_probs/old_probs.detach()
#         obj = ratio*advantage.detach()
#         obj2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantage.detach()
#         clip_loss = torch.mean(torch.min(obj,obj2))
#         loss = -clip_loss + value_coeff*value_loss - entropy_coef*entropy
#         return loss
#
#
# def ppo_update(self, opt, rollout, next_value, max_grad_norm, batch_part=0.5, epochs=10, **args):
#     # loss here is the method of the instance Loss
#     obs, rewards, actions, logits, values, probs = rollout.get()
#     returns = disc_return(rewards, next_value, 0.95) # [length, batch_size if != 1]
#     old_probs = probs.gather(-1, actions).squeeze() # [length, batch_size if != 1]
#     sampler = Sampler(int(batch_part),[obs,actions,returns, old_probs.detach()])
#     for _ in range(epochs):
#         # perform gradient descent for several epochs
#         batch_obs, batch_actions, batch_returns, batch_old_probs = sampler.get_next()
#         batch_logits, batch_values = self.agent.forward(batch_obs)
#         obj = ppo_loss(batch_actions, batch_returns, batch_values.squeeze(), batch_logits, batch_old_probs)
#         obj.backward()
#         torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
#         opt.step()
#         opt.zero_grad()
#     scheduler.step()


class Agent(nn.Module):
    def __init__(self, net, collate_fn, eps=1e-2, device='cuda'):
        super().__init__()
        self.net = net.to(device=device)
        self.collate_fn = collate_fn
        self.device = device
        self.eps = eps
        if self.net.pretrained_path:
            self.net.load_state_dict(torch.load(self.net.pretrained_path))
            if len(self.net.fit_pretrained_layers) > 0:
                self.net.requires_grad_(False)
                unfreezing_p = []
                for n, m in self.net.named_parameters():
                    for l in self.net.fit_pretrained_layers:
                        pattern = re.compile(f"{l}\.|{l}$")
                        if re.match(pattern, n):
                            unfreezing_p.append(n)
                            m.requires_grad_(True)
                LOG.debug(f"Training parameters: {unfreezing_p}")

    def predict(self, inputs, mask):
        # [Nf, max(ni), max(ni)], [Nf, 1]
        logit, values = self.predict_net(self.net, self.collate_fn(inputs))
        dims = logit.shape
        masked_logit = torch.where(mask.view(dims).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device))
        # [Nf, max(ni)*max(ni)]
        probs = masked_logit.view(dims[0], -1).softmax(1).cpu().numpy()
        actions = [np.random.choice(len(prob), p=prob) for prob in probs]
        # convert array of flat indices into a tuple of coordinate arrays
        actions = list(zip(*np.unravel_index(actions, dims[1:])))
        return actions, values.cpu().numpy().squeeze()

    def train_net(self, train_data, epochs, criterion, batch_size, lr, scheduler, gamma, value_loss_coef, entropy_loss_coef, weight_decay, clip_grad_norm, betas, val_data=None, val_steps=100, min_iters=1000):
        LOG.info(f"Start training: {time.ctime()}")
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=lr, betas=betas, weight_decay=weight_decay)

        def lambda_lr(epoch): return scheduler ** np.sqrt(epoch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
        train_dl = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=self.collate_fn, num_workers=0)
        iters = max(min_iters, int(epochs*len(train_dl)))
        di = iter(train_dl)
        for i in range(iters):
            try:
                x_batch, y_batch = next(di)
            except:
                di = iter(train_dl)
                x_batch, y_batch = next(di)
            x_batch, y_batch = to_device(
                x_batch, self.device), to_device(y_batch, self.device)
            pred = self.net(x_batch)
            pg_loss, value_loss, entropy_loss = criterion(
                pred, *y_batch, gamma=gamma)
            (pg_loss + value_loss_coef*value_loss
             + entropy_loss_coef*entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), clip_grad_norm, norm_type=2.0)
            opt.step()
            opt.zero_grad()
            sched.step()

        LOG.info(
            f"""End training: {time.ctime()},
                Policy loss: {pg_loss.item():.2f},
                Value loss: {value_loss.item():.2f},
                Entropy loss {entropy_loss.item():.2f},
                {iters} iterations.""")
        return pg_loss.item(), value_loss.item(), entropy_loss.item()

    def predict_net(self, net, x_batch):
        with evaluation_mode(net):
            x_batch = to_device(x_batch, self.device)
            out = net(x_batch)
        return out


class PolicyGrad():
    def __init__(self, agent, env_config, args, train_args, experience=[], baseline_plans={}):
        self.env_config = env_config
        self.train_args = train_args
        self.env_config['return_latency'] = args['latency']
        self.n_workers = args['n_workers']
        self.total_episodes = args['total_episodes']
        self.n_update = args['n_update']
        self.n_train_episodes = args['n_train_episodes']
        self.gamma = args['gamma']
        self.sync = args['sync']
        self.num_complete_plans = args['num_complete_plans']
        self.save_explored_plans = args['save_explored_plans']
        self.traj_storage = TrajectoryStorage()
        self.experience = PlanExperience(experience, add_sub_plans=False)
        self.agent = agent.share_memory()
        self.agent.eps = args['eps']
        self.step = mp.Value('i', 0)
        self.episode = mp.Value('i', 0)
        self.n_queries = len(env_config['db_data'])
        self.query_ids = mp.Array('i', list(range(self.n_queries)))
        self.update_q = mp.Queue()
        self.step_flag = mp.Event()
        self.baseline_plans = baseline_plans  # {query : plan, ...}
        self.logdir = args['logdir']
        self.cost_func = cost_function[args['cost_func']]
        self.env_config['selectivity'] = args['selectivity']
        self.log_q = mp.Queue()
        encoding = args.get('encoding', 'neo')
        if encoding == 'neo':
            self.env_plan = EnvPlan
        elif encoding == 'rtos':
            self.env_plan = EnvPlanHeapRTOS
        elif encoding == 'neo_pgdata':
            self.env_plan = EnvPlanHeapWithPGdata
        else:
            raise Exception(
                'Wrong encoding name in config. '
                f'Provided "{encoding}" but allowed only "neo" or "rtos"'
            )
        if self.env_config['selectivity']:
            DataBaseEnv_QueryEncoding.compute_cardinalities(self.env_config)

    def run(self):
        runners = [mp.Process(target=self.runner_process, daemon=True)
                   for _ in range(self.n_workers)]
        logger = mp.Process(target=self.logger, daemon=True)
        logger.start()
        LOG.info('Summary writer started.')
        for r in runners:
            r.start()
        self.update_process()

    def logger(self):
        writer = SummaryWriter(self.logdir)
        while self.episode.value < self.total_episodes * self.n_queries:
            r = self.log_q.get()
            if len(r) == 2:
                losses, ep = r
                pg_loss, value_loss, entropy_loss = losses
                writer.add_scalar('Loss/policy_loss', pg_loss, ep)
                writer.add_scalar('Loss/value_loss', value_loss, ep)
                writer.add_scalar('Loss/entropy_loss', entropy_loss, ep)
                torch.save(self.agent.net.state_dict(),
                           Path(self.logdir) / 'state_dict.pt')
            else:
                (n_plans, n_subplans), best_found_costs, generated_costs, baseline_costs, reward, step, episode = r
                writer.add_scalar(
                    'Experience size/complete unique plans', n_plans, episode)
                writer.add_scalar(
                    'Rewards', reward, step)

                for stat_type, costs in (('best_found', best_found_costs), ('generated', generated_costs)):
                    if costs.keys() >= self.env_config['db_data'].keys():
                        writer.add_scalar(f"Cost/{stat_type}/avg_cost:episode",
                                          np.mean(list(costs.values())), episode)
                        if baseline_costs.keys() >= costs.keys():
                            average_ratio = np.mean(
                                [costs[q]/baseline_costs[q] for q in costs])
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:episode', average_ratio, episode)
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:experience', average_ratio, n_plans)

    def runner_process(self):
        env = self.env_plan(self.env_config)
        is_done = True
        while True:
            if is_done:
                if self.episode.value >= self.total_episodes * self.n_queries:
                    return
                with self.episode.get_lock():
                    if self.episode.value % self.n_update == 0 and self.sync:
                        self.step_flag.clear()
                    query_num = self.episode.value % self.n_queries
                    if query_num == 0:
                        np.random.shuffle(self.query_ids)
                    query_idx = self.query_ids[query_num]
                    env.reset(query_idx)
                    self.episode.value += 1
                self.step_flag.wait()
            obs = [env.get_state()]
            mask = get_mask(env.plan)  # [N, max(ni), max(ni)]
            actions, _ = self.agent.predict(obs, mask)
            _, cost, is_done, _ = env.step(actions[0])
            with self.step.get_lock():
                self.step.value += 1
            if is_done:
                self.update_q.put(([env.plan], [cost], env.query_id))

    def update_process(self):
        env = self.env_plan(self.env_config)
        self.traj_storage.set_env(env)
        BASELINE_REWARD = 0.5
        for p in self.baseline_plans.values():
            self.traj_storage.append(p, BASELINE_REWARD)
        generated_costs = {}
        baseline_costs = {q: self.experience.get_cost(
            p, q) for q, p in self.baseline_plans.items()}
        episode = 0
        while True:
            if episode % self.n_update == 0:
                LOG.info(
                    f"Update started, step: {self.step.value}, episode: {episode}, time: {time.ctime()}")
                if episode == 0:
                    data = self.traj_storage.get_dataset(n=self.n_queries)
                else:
                    data = self.traj_storage.get_dataset(
                        n=self.n_train_episodes)
                train_data = data
                val_data = None
                # val_split = max(1, min(self.val_size, int(0.3*len(data))))
                # train_data, val_data = data[:-val_split], data[-val_split:]
                losses = self.agent.train_net(
                    train_data=train_data, val_data=val_data, val_steps=1, criterion=ac_loss, gamma=self.gamma, **self.train_args)
                LOG.info(
                    f"Update ended, step: {self.step.value}, episode: {episode}, time: {time.ctime()}")
                # allow exploring
                self.step_flag.set()
                self.log_q.put((losses, self.step.value))
                # save found plans
                path = Path(self.logdir) / 'plans'
                path.mkdir(parents=True, exist_ok=True)
                best_plans = self.experience.plans_for_queries()
                for q, p in best_plans.items():
                    p.save(path / f"{q}.json")
                LOG.info(
                    f"Best plans after {episode} episodes saved to {str(path)}")

            if (episode >= self.total_episodes * self.n_queries):
                return

            complete_plans, costs, query_id = self.update_q.get()

            for plan, cost in zip(complete_plans, costs):
                # reward = (
                #     baseline_costs[query_id] - cost)/baseline_costs[query_id]
                reward = - np.log(cost/baseline_costs[query_id])
                LOG.debug(
                    f"Completed plan for {query_id} query with cost = {cost}, reward = {reward}")
                self.experience.append(plan, cost, query_id)
                self.traj_storage.append(plan, reward)

            # update values for log
            average_generated_cost = self.experience.get_cost(
                complete_plans[0], query_id)
            if average_generated_cost is not None:
                generated_costs[query_id] = average_generated_cost
            baseline_costs[query_id] = self.experience.get_cost(
                self.baseline_plans[query_id], query_id)
            best_found_costs = self.experience.costs_for_queries()

            self.log_q.put((self.experience.size(), best_found_costs,
                            generated_costs, baseline_costs, reward, self.step.value, episode))

            if self.save_explored_plans and episode % (5 * self.n_queries) == 0:
                for i, (p, q) in enumerate(self.experience.complete_plans.keys()):
                    save_path = Path(self.logdir) / 'all_plans' / str(q)
                    save_path.mkdir(parents=True, exist_ok=True)
                    p.save(save_path / f"{i}.json")

            episode += 1

    def generate_plan_beam_search(self, query_id, num=1):
        is_done = False
        env = self.env_plan(self.env_config, num, False)
        env.reset(query_id)
        self.agent.eps = 0
        i = 0
        while not is_done:
            # print(len(env.plans), i)
            obs = env.get_states()
            valid_actions = env.valid_actions()
            actions = self.agent.predict(obs, valid_actions)
            _, _, is_done, _ = env.step(actions)
            i += 1
        return env.plan


def log_cost(a, *args):
    return np.log(a)


def no_op(a, *args):
    return a


def baseline_ratio_cost(a, baseline, *args):
    return np.array(a)/baseline


def difference_reward(a, baseline, *args):
    return baseline/a - 1.


cost_function = {
    'log': log_cost,
    'no_op': no_op,
    'baseline_ratio': baseline_ratio_cost,
    'difference_reward': difference_reward,
}
