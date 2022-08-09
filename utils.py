# utility code.
from math import sqrt

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import dual_lip_risk_bound as dr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_rho(p_min):
    return 1.0 / p_min


# Dataset container


class train_valid_set:
    def __init__(self, x_in, y_in, x_valid_in, y_valid_rep_in):
        self.x_in = x_in
        self.y_in = y_in
        self.x_valid_in = x_valid_in
        self.y_valid_rep_in = y_valid_rep_in

    def get_reps(self):
        return self.x_valid_in, self.y_valid_rep_in


def split_held(train_len, held_pr):
    indices = np.random.permutation(train_len)
    train_num = int(train_len * (1.0 - held_pr))
    training_idx, test_idx = indices[:train_num], indices[train_num:]
    return training_idx, test_idx


def subarray(array, index):
    return [array[i] for i in index]


class test_set:
    def __init__(self, x_test_in, y_test_rep_in):
        self.x_test_in = x_test_in
        self.y_test_rep_in = y_test_rep_in

    def get_reps(self):
        return self.x_test_in, self.y_test_rep_in


def unroll_reps(x, y_nest):
    x_unr = []
    y_unr = []
    for i in range(len(x)):
        x_unr.extend([x[i]] * len(y_nest[i]))
        y_unr.extend(y_nest[i])
    return np.array(x_unr), y_unr


def reroll_list(unrolled_list, list_struct):
    rolled_list = []
    current_index = 0
    for i in range(len(list_struct)):
        num_to_pop = len(list_struct[i])
        rolled_list.append(unrolled_list[current_index : (current_index + num_to_pop)])
        current_index += num_to_pop
    return rolled_list


def eval_cost(model, loss, x_in, y_nest_in):
    x_rep_in, y_in = unroll_reps(x_in, y_nest_in)
    x = Variable(torch.FloatTensor(x_rep_in), requires_grad=False).to(device)
    y = Variable(
        torch.FloatTensor(np.array(y_in).astype(float))[:, None], requires_grad=False
    ).to(device)
    y_pred = model.forward(x)
    per_ex_loss = loss.forward(y_pred, y)
    return per_ex_loss.data.cpu().numpy()


def eval_cost_single(model, loss, x_in, y_in):
    x = Variable(torch.FloatTensor(x_in), requires_grad=False).to(device)
    y = Variable(
        torch.FloatTensor(np.array(y_in).astype(float))[:, None], requires_grad=False
    ).to(device)
    y_pred = model.forward(x)
    per_ex_loss = loss.forward(y_pred, y)
    return per_ex_loss.data.cpu().numpy()


def get_pred(model, x_in):
    x = Variable(torch.FloatTensor(x_in), requires_grad=False).to(device)
    y_pred = model.forward(x)
    return y_pred.data.cpu().numpy()


# Heldout eval system
def eval_cv_cost(model, test_set_in, loss, baselines, min_pr):
    """
    :param model:
    :param test_set_in:
    :type test_set_in: test_set
    :param loss:
    :return:
    """
    x_test, y_test_rep = test_set_in.get_reps()
    cost = eval_cost(model, loss, x_test, y_test_rep)
    cost_values = reroll_list(cost, y_test_rep)
    # average the values
    avg_cost = [np.mean(cost_set) for cost_set in cost_values]
    avg_cost_sans_baseline = np.array(avg_cost) - np.array(baselines)
    # sort loss
    cost_sorted = np.sort(avg_cost_sans_baseline)[::-1]
    # pick fraction
    sel_num = int(min_pr * len(cost_sorted))
    # reutrn avg.
    return np.mean(cost_sorted[:sel_num]), cost_sorted, avg_cost_sans_baseline


def rad_grid(
    loss,
    robustloss,
    model,
    rho,
    train_set,
    rad_seq,
    niter_inner=500,
    nbisect=10,
    lr=0.01,
    k_dual=2.0,
):
    fv_list = []
    opt_sol_list = []
    x_draw = train_set.x_in
    y_draw = train_set.y_in
    N_tot = x_draw.shape[0]
    liploss = robustloss(0.0, x_draw, np.zeros((N_tot, N_tot)), k_dual=k_dual)
    for rad in tqdm(rad_seq):
        opt_out = dr.opt_model_bisect(
            model,
            loss,
            liploss,
            rho,
            x_draw,
            y_draw,
            rad,
            niter_inner=niter_inner,
            lr=lr,
            nbisect=nbisect,
        )
        fv_list.append(opt_out[0])
        opt_sol_list.append(opt_out[1])
    return fv_list, opt_sol_list


def rad_grid_zo(
    loss,
    robustloss,
    model,
    rho,
    train_set,
    rad_seq,
    niter_inner=500,
    nbisect=10,
    lr=0.01,
):
    fv_list = []
    opt_sol_list = []
    x_draw = train_set.x_in
    y_draw = train_set.y_in
    N_tot = x_draw.shape[0]
    liploss = robustloss(0.0, x_draw, np.zeros((N_tot, N_tot)))
    for rad in tqdm(rad_seq):
        opt_out = dr.opt_model_bisect(
            model,
            loss,
            liploss,
            rho,
            x_draw,
            y_draw,
            rad,
            niter_inner=niter_inner,
            lr=lr,
            nbisect=nbisect,
        )
        fv_list.append(opt_out[0])
        opt_sol_list.append(opt_out[1])
    return fv_list, opt_sol_list


def rad_grid_rkhs(
    loss,
    robustloss,
    model,
    rho,
    train_set,
    rad_seq,
    kern_fn,
    niter_inner=500,
    nbisect=10,
    lr=0.01,
):
    fv_list = []
    opt_sol_list = []
    x_draw = train_set.x_in
    y_draw = train_set.y_in
    N_tot = x_draw.shape[0]
    for rad in tqdm(rad_seq):
        liploss = robustloss(rad, x_draw, np.zeros(N_tot), kern_fn)
        opt_out = dr.opt_model_bisect(
            model,
            loss,
            liploss,
            rho,
            x_draw,
            y_draw,
            rad,
            niter_inner=niter_inner,
            lr=lr,
            nbisect=nbisect,
        )
        fv_list.append(opt_out[0])
        opt_sol_list.append(opt_out[1])
    return fv_list, opt_sol_list
