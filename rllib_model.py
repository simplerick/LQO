import networkx as nx
from PIL import Image
import io
import gym
from gym.spaces import Box

import torch
import numpy as np


#from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomMaskedActModel(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space,
                             num_outputs, model_config,
                             name,
                             **kw)
        torch.nn.Module.__init__(self)
        true_obs_shape=model_config['custom_model_config']['true_obs_shape']
        fc_obs_space = Box(low = 0, high = 1,
                           shape = true_obs_shape,
                           dtype = np.int)
        self.fc_net = FullyConnectedNetwork(fc_obs_space,
                                            action_space,
                                            num_outputs,
                                            model_config,
                                            name + 'fc_net')

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict['obs']['action_mask']

        obs = input_dict['obs']['real_obs'].float()
        obs = obs.reshape(obs.shape[0], -1)
        logits,_ = self.fc_net({'obs': obs})


        inf_mask = torch.clamp(torch.log(action_mask),
                               FLOAT_MIN , FLOAT_MAX)
        return logits+inf_mask, state

    def value_function(self):
        return self.fc_net.value_function()
