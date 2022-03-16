from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from plan import Plan, get_sub_plans
from tree import Tree


def to_forest(plan):
    forest = []
    for root in plan.get_roots():
        g = plan.G.subgraph(nx.descendants(plan.G, root) | {root})
        forest.append(Tree(g, root))
    return forest


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, tuple) or isinstance(obj, list):
        if isinstance(obj[0], float) or isinstance(obj[0], int):
            return obj
        return [to_device(o, device=device) for o in obj]


class Estimate():
    def __init__(self):
        self.t = []

    @property
    def value(self):
        return np.mean(self.t)

    def add(self, value):
        self.t.append(value)

    def new():
        return Estimate()


class PlanExperience():
    """
    Class to store and replenish unique plans and subplans.
    Each subplan S has a value equal the minimum of values of stored complete plans for which S is an actial subplan.
    Can be used with Neo.
    """

    def __init__(self, initial_exp=[], add_sub_plans=True):
        self.complete_plans = defaultdict(
            Estimate.new)  # {(plan1, query): cost1,...}
        self.subplans = defaultdict(
            Estimate.new)
        self.best_plans = dict()  # {query_name : (plan, cost)}
        self.add_sub_plans = add_sub_plans
        for e in initial_exp:
            self.append(*e)

    def size(self):
        return len(self.complete_plans), len(self.subplans)

    def append(self, plan, value, query_id):
        if value is not None:
            estimate = self.complete_plans[(plan, query_id)]
            estimate.add(value)
            if self.add_sub_plans:
                sps = get_sub_plans(plan)
                for sp in sps:
                    sp_est = self.subplans[(sp, query_id)]
                    if len(sp_est.t) == 0 or estimate.value < sp_est.value:
                        self.subplans[(sp, query_id)] = estimate
            if (not query_id in self.best_plans) or (estimate.value < self.best_plans[query_id][1].value):
                self.best_plans[query_id] = (plan, estimate)

    def get_cost(self, plan, query_id):
        est = self.complete_plans.get((plan, query_id))
        return est.value if est else None

    def costs_for_queries(self):
        return {q: x[1].value for q, x in self.best_plans.items()}

    def plans_for_queries(self):
        return {q: x[0] for q, x in self.best_plans.items()}

    def get_dataset(self):
        return [(p, v, q) for (p, q), e in self.subplans.items() for v in e.t]



class PlanExperienceRW(PlanExperience):
    def get_dataset(self):
        ds = []
        for (p, q), e in self.subplans.items():
            ### decreasing weights
            num_step = 2*len(p.rel_leaves)-1 - len(list(p.G.nodes))
            ### increasing weights
            scale = 1/(num_step+1)
            ds.extend((p,scale*v,q) for v in e.t)
        return ds




# context manager for evaluating
@contextmanager
def evaluation_mode(model):
    '''Temporarily switch to evaluation mode. Keeps original training state of every submodule'''
    with torch.no_grad():
        train_state = dict((m, m.training) for m in model.modules())
        try:
            model.eval()
            yield model
        finally:
            # restore initial training state
            for k, v in train_state.items():
                k.training = v


def load_plans_from_dir(path):
    plans = {}
    for p in Path(path).glob("*.json"):
        plan = Plan()
        try:
            plan.load(p)
            plans[p.parts[-1][:-5]] = plan
        except:
            pass
    return plans


def find_next_actions(p, sp):
    actions = set()
    # it's very important that the root node order matches the subtree order in the forest
    enum_roots = {n: i for i, n in enumerate(sp.get_roots())}
    for node in enum_roots:
        for n in p.G.predecessors(node):  # num of predecessors = 0 or 1
            l, r = p.G[n]
            if l in enum_roots and r in enum_roots:
                actions.add((enum_roots[l], enum_roots[r]))
    return actions


def find_inner_join_actions(p):
    actions = []
    roots = p.get_roots()
    for i, n1 in enumerate(roots):
        for j, n2 in enumerate(roots):
            if i != j and p.is_inner_join(n1, n2):
                actions.append((i, j))
    return actions


def get_mask(p):
    size = len(p.get_roots())
    m = torch.zeros((size, size), dtype=bool)
    m[list(zip(*find_inner_join_actions(p)))] = 1
    return m


def stack_masks(masks):
    size = max([mask.shape[0] for mask in masks])
    for i in range(len(masks)):
        l = masks[i].shape[0]
        masks[i] = torch.nn.functional.pad(
            masks[i], (0, size-l, 0, size-l), mode='constant', value=0.0)
    return torch.stack(masks, 0)
