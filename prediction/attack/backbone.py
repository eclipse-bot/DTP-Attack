import numpy as np
import logging
import copy
import torch

from .attack import BaseAttacker
from .loss import attack_loss
from prediction.dataset.generate import input_data_by_attack_step
from .boundary_attack import boundary_attack


logger = logging.getLogger(__name__)


class DTPAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, attack_duration, predictor, n_particles=10, iter_num=30, c1=0.5, c2=0.3, w=1.0, bound=1, physical_bounds={}):
        super().__init__(obs_length, pred_length, attack_duration, predictor)

        self.iter_num = iter_num
        self.bound = bound
        self.physical_bounds = physical_bounds
        self.perturb_length = obs_length + attack_duration - 1
        self.loss = attack_loss
        self.options = {'c1': c1, 'c2': c2, 'w': w}
        self.n_particles = n_particles
        self.bound = bound

    def run(self, data, obj_id, **attack_opts):
        try:
            self.predictor.model.eval()
        except:
            pass

        attack_opts["bound"] = self.bound
        attack_opts["physical_bounds"] = self.physical_bounds

        MSE, loss, perturb, n_calls = boundary_attack(data,obj_id,self.predictor,self.loss,attack_opts,self.obs_length,self.pred_length)
        # repeat the prediction once to get the best output data
        best_out = {}
        perturbation_tensor = torch.from_numpy(perturb[:self.obs_length]).cuda()
        origin_sample = copy.deepcopy(data['objects'][obj_id]['observe_trace'])
        for k in range(self.attack_duration):
            perturbation = {"obj_id": obj_id, "loss": self.loss,"ready_value": {obj_id: torch.zeros(self.obs_length, 2).cuda()},"attack_opts": attack_opts}
            data['objects'][obj_id]['observe_trace'][:self.obs_length] = origin_sample[:self.obs_length,:]+perturb[:self.obs_length]
            input_data = input_data_by_attack_step(data, self.obs_length, self.pred_length, 0)
            output_data, _ = self.predictor.run(input_data, perturbation=perturbation, backward=False)
            output_data['objects'][obj_id]['observe_trace'][:self.obs_length] = origin_sample[:self.obs_length]
            best_out[str(k)] = output_data
        return {
            "output_data": best_out,
            "perturbation": {obj_id: perturbation_tensor[:self.obs_length].cpu().detach().numpy()},
            "loss":n_calls,
            "obj_id": obj_id,
            "attack_opts": attack_opts,
            "attack_length": self.attack_duration
        }
