import numpy as np
import copy
import time
import torch
from prediction.dataset.generate import input_data_by_attack_step

def get_diff(sample_1, sample_2):
	"""Channel-wise norm of difference between samples."""
	return np.linalg.norm(sample_1 - sample_2, axis=1)
def isadversarial(loss):
	return loss<-60
def forward_perturbation(epsilon, prev_sample, target_sample):
	"""Generate forward perturbation."""
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb *= epsilon
	return perturb

def orthogonal_perturbation(delta, prev_sample, target_sample,obs_length,pre_length):
    """Generate orthogonal perturbation."""
    perturb = np.random.randn(obs_length, 2)
    for i in range(obs_length):
	    perturb[i,:]/=np.linalg.norm(perturb, axis = 1)[i]
    perturb *= delta * np.mean(get_diff(target_sample[:obs_length,:], prev_sample[:obs_length,:]))
	# Project perturbation onto sphere around target
    diff = (target_sample[:obs_length,:] - prev_sample[:obs_length,:]).astype(np.float32) # Orthorgonal vector to sphere surface
    for i in range(obs_length):
	    diff[i,:]/=np.linalg.norm(diff,axis = 1)[i] # Orthogonal unit vector
	# We project onto the orthogonal then subtract from perturb
	# to get projection onto sphere surface
    perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff
    for i in range(obs_length):
        for j in range(2):
            if perturb[i,j] > 2:
                perturb[i,j] = 2
            if perturb[i,j] < -2:
                perturb[i,j] = -2;
    a = np.zeros(prev_sample.shape)
    a[:obs_length,:] = perturb
    return a

def boundary_attack(data, obj_id, predictor, loss_func, attack_opts, obs_length, pre_length):
    p0 = {"obj_id": obj_id, "loss": loss_func, "ready_value": {obj_id: torch.zeros(obs_length, 2).cuda()},"attack_opts": attack_opts}
    target_sample = copy.deepcopy(data['objects'][obj_id]['observe_trace'])
    while True:
        perturbation = np.random.uniform(-10, 10, size = (1, 2*obs_length))
        perturbation = perturbation.reshape((obs_length, 2))
        initial_sample = copy.deepcopy(data['objects'][obj_id]['observe_trace'])
        initial_sample[:obs_length] = initial_sample[:obs_length]+perturbation
        data['objects'][obj_id]['observe_trace'] = initial_sample
        input_data = input_data_by_attack_step(data, obs_length, pre_length, 0)
        _, _loss = predictor.run(input_data, perturbation=p0, backward=False)
        if(isadversarial(_loss)):
            print("finded initial adversarial sample")
            break

    adversarial_sample = initial_sample
    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.1
    global_MSE = 10000
    # Move first step to the boundary
    i = 0
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
        data['objects'][obj_id]['observe_trace'] = trial_sample
        input_data = input_data_by_attack_step(data, obs_length, pre_length, 0)
        _,_loss = predictor.run(input_data,perturbation=p0,backward=False)
        n_calls += 1
        if isadversarial(_loss):
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9
	# Iteratively run attack
    while True:
        print("Step #{}...".format(n_steps))
        # Orthogonal step
        print("\tDelta step...")
        d_step = 0
        while True:
            d_step += 1
            print("\t#{}".format(d_step))
            trial_samples = []
            for i in np.arange(10):
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample, obs_length, pre_length)
                trial_samples.append(trial_sample)
            result = []
            for i in trial_samples:
                data['objects'][obj_id]['observe_trace']=i;
                input_data = input_data_by_attack_step(data, obs_length, pre_length, 0)
                _, _loss = predictor.run(input_data, perturbation=p0, backward=False)
                if(isadversarial(_loss)):
                    result.append(1)
                else:
                    result.append(0)
            result = np.array(result)
            n_calls += 10
            d_score = np.mean(result)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.95
                elif d_score > 0.7:
                    delta /= 0.95
                adversarial_sample = trial_samples[np.where(result == 1)[0][0]]
                break
            else:
                delta *= 0.95
        # Forward step
        print("\tEpsilon step...")
        e_step = 0
        while True:
            e_step += 1
            print("\t#{}".format(e_step))
            trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
            data['objects'][obj_id]['observe_trace'] = trial_sample
            input_data = input_data_by_attack_step(data, obs_length, pre_length, 0)
            _, _loss = predictor.run(input_data, perturbation=p0, backward=False)
            n_calls += 1
            if(isadversarial(_loss)):
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            elif e_step > 500:
                break
            else:
                epsilon *= 0.5

        n_steps += 1
        chkpts = [1, 5, 10, 50, 100, 500]
        if (n_steps in chkpts) or (n_steps % 500 == 0):
            print("{} steps".format(n_steps))
        diff = np.mean(get_diff(adversarial_sample, target_sample)[:obs_length])
        if diff<global_MSE:
            global_MSE = diff
            global_adver = adversarial_sample
            global_loss = _loss
        if diff <= 1e-3 or e_step > 500 or diff > 30 or n_steps > 100:#n_steps>100 n_calls > 2000
            print("{} steps".format(n_steps))
            print("Mean Squared Error: {}".format(diff))
            print("Global_MSE,loss: {},{}".format(global_MSE,global_loss))
            print("Adversarial_sample: {}".format(global_adver))
            data['objects'][obj_id]['observe_trace'] = target_sample
            return global_MSE, global_loss, global_adver-target_sample, n_calls

        print("Mean Squared Error: {}".format(diff))
        print("Calls: {}".format(n_calls))
