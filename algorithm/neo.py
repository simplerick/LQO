from plan import *
from database_env import DataBaseEnv_QueryEncoding, DataBaseEnv_RTOSEncoding
from utils.utils import *
from utils.db_utils import *
from tree import *
from heapq import *
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter
import re
import networkx as nx

import logging
LOG = logging.getLogger(__name__)


class EnvBase(DataBaseEnv_QueryEncoding):
    def __init__(self, env_config):
        super().__init__(env_config)

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


class EnvTreeSearch(EnvBase):
    def reset(self, idx=None):
        super().reset(idx)
        self._graph = nx.DiGraph()
        self.add_node(self.plan)

    def node_attrs(self, node):
        return self._graph.nodes[node]

    def add_node(self, plan, value=None):
        id = len(self._graph)
        self._graph.add_node(id, plan=plan, value=value, counter=0)
        return id

    def add_edge(self, n1, n2, join):
        self._graph.add_edge(n1, n2, join=join)

    def get_path(self, n1, n2):
        n_max = max(n1, n2)
        n_min = min(n1, n2)
        path = [n_max]
        while path[-1] != n_min:
            path.append(next(self._graph.predecessors(path[-1])))
        return path[::-1]

    def expand(self, node):
        plans = get_sup_plans(self._graph.nodes[node]["plan"])
        nodes = []
        for p in plans:
            n = self.add_node(p)
            nodes.append(n)
            self.add_edge(node, n, list(p.G[p._last_join_node]))
        return nodes

    def assign(self, nodes, **attrs):
        for i, n in enumerate(nodes):
            for attr, values in attrs.items():
                self._graph.nodes[n][attr] = values[i]


class EnvDepthSearch(EnvTreeSearch):
    def __init__(self, env_config, track_true_reward=True, planning_depth=2):
        super().__init__(env_config)
        self.planning_depth = planning_depth
        self.track_true_reward = track_true_reward

    def reset(self, idx):
        super().reset(idx)
        self._node = 0
        self.max_depth = len(self.plan.alias_to_table)-1
        self._depth = 0
        self.costs = []
        self.complete_plans = []

    def valid_actions(self):
        self._nodes = [self._node]
        for step in range(min(self.planning_depth, self.max_depth-self._depth)):
            new_level = []
            for n in self._nodes:
                new_level.extend(self.expand(n))
            self._nodes = new_level
        obs = []
        for n in self._nodes:
            obs.append(self.get_obs(self.node_attrs(n)["plan"]))
        return obs

    def step(self, values):
        min_ix = np.argmin(values)
        self._node = self.get_path(self._node, self._nodes[min_ix])[1]
        self._depth += 1
        self.plan = self.node_attrs(self._node)["plan"]
        is_complete = self.plan.is_complete
        if is_complete:
            self.complete_plans.append(self.plan)
            if self.track_true_reward:
                self.costs.append(self.reward())
        return self.complete_plans, self.costs, is_complete, None


class EnvPlanHeap(EnvBase):
    def __init__(self, env_config, max_heap_size=500, track_true_reward=True):
        super().__init__(env_config)
        self.max_heap_size = max_heap_size
        self.track_true_reward = track_true_reward

    def reset(self, idx=None):
        self.min_heap = []
        self.complete_plans = []
        self.costs = []
        return super().reset(idx)

    def valid_actions(self):
        self._actions = get_sup_plans(self.plan)
        obs = []
        for p in self._actions:
            obs.append(self.get_obs(p))
        return obs

    def step(self, action, values):
        is_complete = next(iter(self._actions)).is_complete
        if is_complete:
            self.plan = self._actions[np.argmin(values)]
            self.complete_plans.append(self.plan)
            if self.track_true_reward:
                cost = self.reward()
                self.costs.append(cost)
        else:
            for c, p in zip(values, self._actions):
                heappush(self.min_heap, (c, p))
        if len(self.min_heap) == 0:
            return None, None, is_complete, True

        if not is_complete and action is not None:
            self.plan = self._actions[action]
        elif (len(self.min_heap) < self.max_heap_size or is_complete):
            # heap mode
            self.plan = heappop(self.min_heap)[1]
        else:
            # greedy mode
            self.plan = self._actions[np.argmin(values)]
        return None, None, is_complete, False


class EnvPlanBeam(EnvPlanHeap):
    def __init__(self, env_config, num_complete_plans, track_true_reward=True):
        super().__init__(env_config)
        self.num_complete_plans = num_complete_plans
        self.selectivity = env_config['selectivity']
        self.track_true_reward = track_true_reward

    def reset(self, idx=None):
        super().reset(idx)
        self.plans = [self.plan]
        self.costs = []
        return self.get_obs()

    def valid_actions(self):
        actions = set()
        for plan in self.plans:
            for sp in get_sup_plans(plan):
                actions.add(sp)
        self._actions = list(actions)
        obs = []
        for p in self._actions:
            self.plan = p
            obs.append(self.get_obs())
        return obs

    def step(self, action, values):
        is_complete = next(iter(self._actions)).is_complete
        ids = np.argsort(values)[:self.num_complete_plans]
        self.plans = [self._actions[i] for i in ids]
        if is_complete and self.track_true_reward:
            for plan in self.plans:
                self.plan = plan
                self.costs.append(self.reward())
        return None, None, is_complete, None


class EnvPlanHeapRTOS(EnvPlanHeap, DataBaseEnv_RTOSEncoding):
    def compute_query_enc(self):
        self.column_represenation = self.get_column_representation()
        self.join_graph_encoding = self.get_join_graph_encoding()

    def get_obs(self):
        self.apply_plan_encoding()
        return (
            np.concatenate([
                self.join_graph_encoding,
                self.column_represenation.flatten()
            ]).astype(np.int),
            to_forest(self.plan)
        )


def collate(batch):
    if isinstance(batch[0][0], np.ndarray):
        q_enc, forests = zip(*batch)
        q_enc = np.asarray(q_enc)
        lens = [len(f) for f in forests]
        flatten_tc_batch = flatten_batch_TreeConv([t.to_torch().to(
            dtype=torch.float) for f in forests for t in f], batch_first=False)
        q_enc = torch.repeat_interleave(torch.tensor(
            q_enc, dtype=torch.float), torch.tensor(lens), dim=0)
        # [n_sum, d_qenc], ([L,n_sum,d_enc],[3L,n_sum,1],[batch_size])
        return q_enc, (*flatten_tc_batch, lens)
    x, y = zip(*batch)
    return collate(x), torch.tensor(y, dtype=torch.float)


class Agent(nn.Module):
    def __init__(self, net, collate_fn, eps=0, device='cuda'):
        super().__init__()
        self.net = net.to(device=device)
        self.collate_fn = collate_fn
        self.device = device
        self.eps = eps

    def predict(self, inputs):
        out = self.predict_net(self.net, inputs, batch_size=100).cpu().numpy()
        if np.random.random() < self.eps:
            action = np.random.choice(len(inputs))
        else:
            action = None
        return action, out

    def train_net(self, train_data, epochs, criterion, batch_size, lr, scheduler, betas, val_data=None, val_steps=100, min_iters=1000):
        LOG.info(f"Start training: {time.ctime()}")
        net = self.net.new().to(device=self.device)
        net.train()
        if self.net.pretrained_path:
            net.requires_grad_(False)
            unfreezing_p = []
            for n, m in net.named_parameters():
                for l in net.fit_pretrained_layers:
                    pattern = re.compile(f"{l}\.|{l}$")
                    if re.match(pattern, n):
                        unfreezing_p.append(n)
                        m.requires_grad_(True)
            LOG.debug(f"Training parameters: {unfreezing_p}")
        opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[200, 500, 1000, 1500, 2000, 2500, 3000, 3500], gamma=scheduler)
        min_val_loss = np.inf
        min_train_loss = np.inf
        train_dl = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=self.collate_fn, num_workers=0)
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
            pred = net(x_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            if i % val_steps == 0:
                if loss.item() < min_train_loss:
                    min_train_loss = loss.item()
                    if val_data is None:
                        state_dict = deepcopy(net.state_dict())
                LOG.debug(f'train: {loss.item()}, step: {i}')
                if val_data is not None:
                    val_loss = self.evaluate_net(
                        net, criterion, batch_size, val_data)
                    LOG.debug(f'validation: {val_loss}, step: {i}')
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        state_dict = deepcopy(net.state_dict())
        self.net.load_state_dict(state_dict)
        LOG.info(
            f"End training: {time.ctime()}, {min_train_loss:.2f}, {min_val_loss:.2f}, {iters} iterations.")
        return min_train_loss, min_val_loss

    def predict_net(self, net, inputs, batch_size):
        dl = torch.utils.data.DataLoader(
            inputs, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=0)
        preds = []
        with evaluation_mode(net):
            for x_batch in dl:
                x_batch = to_device(x_batch, self.device)
                preds.append(net(x_batch).view(-1))
        return torch.cat(preds)

    def evaluate_net(self, net, criterion, batch_size, val_data):
        preds = self.predict_net(net, [v[0] for v in val_data], batch_size)
        gt = torch.tensor([v[1] for v in val_data], device=self.device)
        loss = criterion(preds, gt)
        return loss.item()


class Neo():
    def __init__(self, agent, env_config, args, train_args, experience=[], baseline_plans={}):
        self.env_config = env_config
        self.train_args = train_args
        self.env_config['return_latency'] = args['latency']
        self.n_workers = args['n_workers']
        self.total_episodes = args['total_episodes']
        self.sync = args['sync']
        self.val_size = args['val_size']
        self.num_complete_plans = args['num_complete_plans']
        if args.get('reward_weighting'):
            self.experience = PlanExperienceRW(experience)
        else:
            self.experience = PlanExperience(experience)
        self.agent = agent.share_memory()
        self.agent.eps = args['eps']
        self.step = mp.Value('i', 0)
        self.episode = mp.Value('i', 0)
        self.n_queries = len(env_config['db_data'])
        self.random_query_ids = mp.Array('i', list(range(self.n_queries)))
        self.update_q = mp.Queue()
        self.step_flag = mp.Event()
        self.baseline_plans = baseline_plans  # {query : plan, ...}
        self.logdir = args['logdir']
        self.cost_func = cost_function[args['cost_func']]
        self.log_q = mp.Queue()
        encoding = args.get('encoding', 'neo')
        selectivity = args.get('selectivity', False)
        cardinality = args.get('cardinality', False)
        if encoding == 'neo_pgdata':
            DataBaseEnv_QueryEncoding.compute_data_driven_features(
                self.env_config)
        if selectivity and 'condition_selectivity' not in env_config:
            DataBaseEnv_QueryEncoding.compute_condition_selectivities(self.env_config)
        if cardinality and 'cardinalities' not in env_config:
            DataBaseEnv_QueryEncoding.compute_cardinalities(self.env_config)
        self.env_plan_heap = EnvPlanHeap

    def run(self):
        runners = [mp.Process(target=self.runner_process)
                   for _ in range(self.n_workers)]
        logger = mp.Process(target=self.logger)
        logger.start()
        for r in runners:
            r.start()
        self.update_process()
        for r in runners:
            r.terminate()
        logger.terminate()

    def logger(self):
        writer = SummaryWriter(self.logdir)
        while self.episode.value < self.total_episodes * self.n_queries:
            r = self.log_q.get()
            if len(r) == 2:
                losses, ep = r
                torch.save(self.agent.net.state_dict(),
                           Path(self.logdir) / 'state_dict.pt')
                writer.add_scalar('Loss/train', losses[0], ep)
                writer.add_scalar('Loss/val', losses[1], ep)
            else:
                (n_plans, n_subplans), best_found_costs, generated_costs, baseline_costs, heap_size, step, episode = r
                writer.add_scalar(
                    'Experience size/complete unique plans', n_plans, episode)
                writer.add_scalar(
                    'Experience size/unique sublans', n_subplans, episode)
                writer.add_scalar('Heap size', heap_size, episode)

                for stat_type, costs in (('best_found', best_found_costs), ('generated', generated_costs)):
                    if costs.keys() >= self.env_config['db_data'].keys():
                        writer.add_scalar(f"Cost/{stat_type}/avg_cost:episode",
                                          np.mean(list(costs.values())), episode)
                        if baseline_costs.keys() >= costs.keys():
                            average_ratio = np.mean(
                                [costs[q]/baseline_costs[q] for q in costs])
                            # writer.add_scalar(f'Cost/{stat_type}/avg_baseline_ratio:step', average_ratio, step)
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:episode', average_ratio, episode)
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:experience', average_ratio, n_plans)

    def runner_process(self):
        env = self.env_plan_heap(self.env_config)
        exhausted = True
        while True:
            if (exhausted or len(env.complete_plans) == self.num_complete_plans):
                if self.episode.value >= self.total_episodes * self.n_queries:
                    return
                with self.episode.get_lock():
                    query_num = self.episode.value % self.n_queries
                    if query_num == 0:
                        if self.sync:
                            self.step_flag.clear()
                        np.random.shuffle(self.random_query_ids)
                    randomized_query_idx = self.random_query_ids[query_num]
                    env.reset(randomized_query_idx)
                    self.episode.value += 1
                self.step_flag.wait()
            sup_plans = env.valid_actions()
            _, _, _, exhausted = env.step(*self.agent.predict(sup_plans))
            with self.step.get_lock():
                self.step.value += 1
            if (exhausted or len(env.complete_plans) == self.num_complete_plans):
                LOG.debug(
                    f"Completed plans for {env.query_id} query with costs = {env.costs}")
                self.update_q.put(
                    (env.complete_plans, env.costs, len(env.min_heap), env.query_id))

    def update_process(self):
        env = self.env_plan_heap(self.env_config)
        generated_costs = {}
        baseline_costs = {q: self.experience.get_cost(
            p, q) for q, p in self.baseline_plans.items()}
        episode = 0
        while True:
            if episode % self.n_queries == 0:
                LOG.info(
                    f"Update started, step: {self.step.value}, episode: {episode}, time: {time.ctime()}")
                data = self.experience.get_dataset()
                # compute features
                for i in range(len(data)):
                    data[i] = (env.get_obs(data[i][0]), self.cost_func(
                        data[i][1], baseline_costs[data[i][2]]))
                np.random.shuffle(data)
                val_split = max(1, min(self.val_size, int(0.3*len(data))))
                train_data, val_data = data[:-val_split], data[-val_split:]
                losses = self.agent.train_net(
                    train_data=train_data, val_data=val_data, val_steps=200, criterion=nn.MSELoss(), **self.train_args)
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

            complete_plans, costs, heap_size, query_id = self.update_q.get()
            for plan, cost in zip(complete_plans, costs):
                self.experience.append(plan, cost, query_id)
            # update values for log
            average_generated_cost = self.experience.get_cost(
                complete_plans[0], query_id)
            if average_generated_cost is not None:
                generated_costs[query_id] = average_generated_cost
            baseline_costs[query_id] = self.experience.get_cost(
                self.baseline_plans[query_id], query_id)
            best_found_costs = self.experience.costs_for_queries()
            self.log_q.put((self.experience.size(), best_found_costs, generated_costs,
                            baseline_costs, heap_size, self.step.value, episode))

            if episode % (5 * self.n_queries) == 0:
                save_path = Path(self.logdir) / 'all_plans'
                save_path.mkdir(parents=True, exist_ok=True)
                for i, (p, q) in enumerate(self.experience.complete_plans.keys()):
                    p.save(save_path / f"{i}_{q}.json")

            episode += 1

    def generate_plan(self, query_id, num=1):
        exhausted = False
        env = self.env_plan_heap(self.env_config, track_true_reward=False)
        env.reset(query_id)
        costs = []
        self.agent.eps = 0
        while not exhausted and len(env.complete_plans) < num:
            sup_plans = env.valid_actions()
            preds = self.agent.predict(sup_plans)
            _, _, is_complete, exhausted = env.step(*preds)
            if is_complete:
                costs.append(preds[1][0])
        return env.complete_plans[np.argmin(costs)]

    def generate_plan_beam_search(self, query_id, num=1):
        is_done = False
        self.agent.eps = 0
        env = EnvPlanBeam(self.env_config, num, track_true_reward=False)
        env.reset(query_id)
        while not is_done:
            sup_plans = env.valid_actions()
            _, _, is_done, _ = env.step(*self.agent.predict(sup_plans))
        return env.plans[0]


def log_cost(a, *args):
    return np.log(a)


def no_op(a, *args):
    return a


def baseline_ratio_cost(a, baseline, *args):
    return a/baseline


cost_function = {
    'log': log_cost,
    'no_op': no_op,
    'baseline_ratio': baseline_ratio_cost
}
