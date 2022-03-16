from plan import *
from database_env import *
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


class EnvPlanBeam(DataBaseEnv_QueryEncoding):
    def __init__(self, env_config, num_complete_plans):
        super().__init__(env_config)
        self.num_complete_plans = num_complete_plans

    def reset(self, idx=None):
        super().reset(idx)
        self.plans = [self.plan]
        self.costs = []
        return self.get_obs()

    def compute_query_enc(self):
        self.predicate_ohe = self.get_predicates_ohe()
        self.join_graph_encoding = self.get_join_graph_encoding()

    def apply_plan_encoding(self):
        g = self.plan.G
        for n in g:
            if not "feature" in g.nodes[n]:
                # ONEHOT
                # vec = np.zeros(len(self.rels))
                # vec[[self.rel_to_idx[t] for t in g.nodes[n]['tables']]] = 1

                # SELECTIVITY INSTEAD 1
                # vec = np.zeros(len(self.rels))
                # d = defaultdict(list)
                # for te in g.nodes[n]['tab_entries']:
                #     d[self.rel_to_idx[self.plan.alias_to_table[te]]].append(self.db_data[self.query_id][4][te])
                # for t, sels in d.items():
                #     vec[t] = np.mean(sels)

                # SELECTIVITY ADDITIONAL DIM
                vec = np.zeros(len(self.rels)+1)
                vec[[self.rel_to_idx[t] for t in g.nodes[n]['tables']]] = 1
                vec[-1] = np.mean([self.db_data[self.query_id][4][te] for te in g.nodes[n]['tab_entries']])

                g.nodes[n]["feature"] = vec.reshape(1,-1)

    def get_obs(self):
        self.apply_plan_encoding()
        return np.concatenate([self.join_graph_encoding, self.predicate_ohe]), to_forest(self.plan)

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


    def step(self, values):
        is_complete = next(iter(self._actions)).is_complete
        ids = np.argsort(values)[:self.num_complete_plans]
        self.plans  = [self._actions[i] for i in ids]
        if is_complete:
            for plan in self.plans:
                self.plan = plan
                self.costs.append(self.reward())
        return None, None, is_complete, None





class Net(nn.Module):
    def __init__(self, d_emb, d_query, d_model, nhead, ffdim, nlayers, fc_nlayers, drop, **kwargs):
        super().__init__()
        self.args = {k: v for k,v in locals().items() if k not in ['self', '__class__']}
        # Tree transformer
        self.enc = nn.Linear(d_emb, (d_model+1)//2)
        self.trans_enc = nn.TransformerEncoder(TreeTransformerEncoderLayer(d_model, nhead, ffdim, drop), nlayers)
        self.cls = nn.Parameter(torch.empty((1, 1, d_model)))
        torch.nn.init.xavier_uniform_(self.cls, gain=1.0)
        # Transformer encoder for combining forest repr (bunch of vectors) into one vector
        self.many_to_one = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 1, ffdim, drop), 1)
        self.cls2 = nn.Parameter(torch.empty((1, 1, d_model)))
        torch.nn.init.xavier_uniform_(self.cls2, gain=1.0)
        # FC layers
        dims = torch.linspace(d_model, 1, fc_nlayers+1, dtype=torch.long)
        layers = []
        for i in range(fc_nlayers-1):
            layers.extend([nn.Dropout(drop), nn.Linear(int(dims[i]), int(dims[i+1])), nn.BatchNorm1d(int(dims[i+1])), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], 1))
        self.fc = nn.Sequential(*layers)
        # Query level
        d_q = d_model // 2
        self.qn = nn.Sequential(nn.Linear(d_query, d_q))

    def forward(self, inputs):
        q, t = inputs
        q = self.qn(q).unsqueeze(0) # [1, (n1+n2+...), d_model // 2]
        x, indices, lens = t
        x = self.enc(x) # [L, (n1+n2+...), d_model // 2]; ni = number of trees in i-th forest
        x = torch.cat((x, q.expand(x.shape[0], -1, -1)), -1) # [L, (n1+n2+...), d_model]
        x = torch.cat((self.cls.expand(-1,x.shape[1],-1), x), 0)
        x, _ = self.trans_enc((x, indices)) # [1, (n1+n2+...), d_model], ...
        l = torch.split(x[0], lens)
        x = torch.nn.utils.rnn.pad_sequence(l) # [max(ni), Nf, d_model]
        x = torch.cat((self.cls2.expand(1,x.shape[1],-1), x), 0)
        pad_mask = torch.tensor(np.arange(x.shape[0]).reshape(1,-1) > np.array(lens).reshape(-1,1), device=x.device)
        x = self.many_to_one(x, src_key_padding_mask=pad_mask)[0] # [N_f, d_model]
        x = self.fc(x)
        return x

    def new(self):
        return self.__class__(**self.args)




def collate(batch):
    if isinstance(batch[0][0], np.ndarray):
        q_enc, forests = zip(*batch)
        lens = [len(f) for f in forests]
        flatten_tc_batch = flatten_batch_TreeConv([t.to_torch().to(dtype=torch.float) for f in forests for t in f], batch_first = False)
        q_enc = torch.repeat_interleave(torch.tensor(q_enc, dtype=torch.float), torch.tensor(lens), dim=0)
        return q_enc, (*flatten_tc_batch, lens) # [n_sum, d_qenc], ([L,n_sum,d_enc],[3L,n_sum,1],[batch_size])
    x, y = zip(*batch)
    return collate(x), torch.tensor(y, dtype=torch.float)






class Agent(nn.Module):
    def __init__(self, net, collate_fn, eps=1e-2, device = 'cuda'):
        super().__init__()
        self.net = net.to(device=device)
        self.collate_fn = collate_fn
        self.device = device
        self.eps = eps

    def predict(self, inputs):
        return self.predict_net(self.net, inputs, batch_size=100).cpu().numpy()

    def train_net(self, train_data, iters, criterion, batch_size, lr, scheduler, betas, val_data=None, val_steps = 100):
        net = self.net.new().to(device=self.device)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[200, 500, 1000, 1500, 2000, 2500], gamma=scheduler)
        losses = [np.inf]
        train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=self.collate_fn, num_workers=0)
        iters = max(len(train_dl), iters)
        di = iter(train_dl)
        print("Start training: ", time.ctime())
        for i in range(iters):
            try:
                x_batch, y_batch = next(di)
            except:
                di = iter(train_dl)
                x_batch, y_batch = next(di)
            x_batch, y_batch = to_device(x_batch, self.device), to_device(y_batch, self.device)
            pred = net(x_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            if i % val_steps == 0:
                if val_data is not None:
                    losses.append(self.evaluate_net(net, criterion, batch_size, val_data))
                else:
                    losses.append(loss.item())
                if losses[-1] == np.min(losses):
                    state_dict = deepcopy(net.state_dict())
                print('train: ', loss.item())
                print('validation: ', losses[-1], ', step: ', i)
        self.net.load_state_dict(state_dict)
        print("End training: ", time.ctime())
        return losses[1:]

    def predict_net(self, net, inputs, batch_size):
        dl = torch.utils.data.DataLoader(inputs, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=0)
        preds = []
        with evaluation_mode(net):
            for x_batch in dl:
                x_batch = to_device(x_batch, self.device)
                preds.append(net(x_batch).view(-1))
        return torch.cat(preds)

    def evaluate_net(self, net, criterion, batch_size, val_data):
        preds = self.predict_net(net, [v[0] for v in val_data], batch_size)
        gt = torch.tensor([v[-1] for v in val_data], device=self.device)
        loss = criterion(preds, gt)
        return loss.item()





class NeoBeam():
    def __init__(self, agent, env_config, args, train_args, experience = [], baseline_costs=None):
        self.env_config = env_config
        self.train_args = train_args
        self.env_config['return_latency'] = args['latency']
        self.n_workers = args['n_workers']
        self.total_episodes = args['total_episodes']
        self.sync = args['sync']
        self.val_size = args['val_size']
        self.num_complete_plans = args['num_complete_plans']
        self.experience = PlanExperience(experience)
        self.agent = agent.share_memory()
        self.step = mp.Value('i', 0)
        self.episode = mp.Value('i', 0)
        self.n_queries = len(env_config['db_data'])
        self.update_q = mp.Queue()
        self.step_flag = mp.Event()
        self.baseline_costs = baseline_costs
        self.logdir = args['logdir']
        self.log_q = mp.Queue()

    def run(self):
        runners = [mp.Process(target=self.runner_process) for _ in range(self.n_workers)]
        logger = mp.Process(target=self.logger)
        logger.start()
        for r in runners:
            r.start()
        self.update_process()


    def logger(self):
        writer = SummaryWriter(self.logdir)
        while self.step.value < self.total_steps:
            r = self.log_q.get()
            if len(r) == 2:
                writer.add_scalar('Loss/val', r[0], r[1])
                torch.save(self.agent.net.state_dict(), Path(self.logdir) / 'state_dict.pt')
            else:
                cost, (n_plans, n_subplans), costs, heap_size, step, episode  = r
                average_cost = np.mean(list(costs.values()))
                average_log_cost = np.mean(np.log(list(costs.values())))
                writer.add_scalar('Cost/log(cost):step', np.log(cost), step)
                writer.add_scalar('Cost/average_cost:step', average_cost, step)
                writer.add_scalar('Cost/average_cost:experience', average_cost, n_plans)
                writer.add_scalar('Cost/average_log(cost):step', average_log_cost, step)
                writer.add_scalar('Cost/average_log(cost):experience', average_log_cost, n_plans)
                if self.baseline_costs is not None and self.baseline_costs.keys() == costs.keys():
                    average_ratio = np.mean([costs[q]/self.baseline_costs[q] for q in costs])
                    writer.add_scalar('Cost/average_baseline_ratio:step', average_ratio, step)
                    writer.add_scalar('Cost/average_baseline_ratio:episode', average_ratio, episode)
                    writer.add_scalar('Cost/average_baseline_ratio:experience', average_ratio, n_plans)
                writer.add_scalar('Experience size/complete unique plans', n_plans, step)
                writer.add_scalar('Experience size/unique sublans', n_subplans, step)


    def runner_process(self):
        env = EnvPlanBeam(self.env_config)
        is_done = True
        while self.step.value < self.total_steps:
            if is_done:
                with self.episode.get_lock():
                    query_num = self.episode.value % self.n_queries
                    env.reset(query_num)
                    self.episode.value += 1
                    if query_num == 0:
                        self.update_q.put(('update', self.step.value, self.episode.value))
                        if self.sync:
                            self.step_flag.clear()
            self.step_flag.wait()
            sup_plans = env.valid_actions()
            _, _, is_done, _ = env.step(self.agent.predict(sup_plans))
            with self.step.get_lock():
                self.step.value += self.num_complete_plans
                if self.step.value == self.total_steps:
                    self.update_q.put(('update', self.step.value, self.episode.value))
            if is_done:
                self.update_q.put(('store', env.plans, env.costs, None, env.query_id))



    def update_process(self):
        env = EnvPlanBeam(self.env_config, self.num_complete_plans)
        while self.step.value < self.total_steps:
            c, *r = self.update_q.get()
            if c == 'store':
                complete_plans, costs, heap_size, query_id = r
                for plan, cost in zip(complete_plans, costs):
                    self.experience.append(plan, cost, query_id)
                self.log_q.put((np.mean(costs), self.experience.size(), self.experience.costs_for_queries(), heap_size, self.step.value, self.episode.value))
            else:
                step, episode = r
                print(f"Update started, step: {step}, episode: {episode}, time: {time.ctime()}")
                data = self.experience.get_dataset()
                # compute features
                for i in range(len(data)):
                    env.plan = data[i][0]
                    env.compute_query_enc()
                    data[i] = (env.get_obs(), cost_function(data[i][1]))
                val_split = max(1, min(5000, int(0.3*len(data))))
                train_data, val_data = data[:-val_split], data[-val_split:]
                losses = self.agent.train_net(train_data=train_data, val_data=val_data, val_steps=200, criterion=nn.MSELoss(), **self.train_args)
                self.step_flag.set()
                self.log_q.put((np.min(losses), step))
                # save found plans
                path = Path(self.logdir) / 'plans'
                path.mkdir(parents=True, exist_ok=True)
                best_plans = self.experience.plans_for_queries()
                for q, p in best_plans.items():
                    p.save(path / f"{q}.json")

    def generate_plan(self, query_id, num=10):
        is_done = False
        env = EnvPlanBeam(self.env_config, num)
        env.reset(query_id)
        while not is_done:
            sup_plans = env.valid_actions()
            _, _, is_done, _ = env.step(self.agent.predict(sup_plans))
        return env.plans[0]





def cost_function(a):
    return np.log(a)
