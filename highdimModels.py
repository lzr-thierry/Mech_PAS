import sys, os
import numpy as np
import torch
from FNN_surrogate_nested import Surrogate
from utils import load_st_idt
import glob
import re
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


def build_net(ckpt, device):
    in_dim = len(ckpt['red_rxn_idx'])
    net = nn.Sequential(
        nn.Linear(in_dim, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net


class Highdim:
    def __init__(self):
        # Init parameters
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.mech_path = './nh3_mech/NH3_Zhang2023_KAUST.yaml'
        ckpt_dir ='./sl_logs/N_mc100000'
        st_idt_paths = ['./exp_data/nh3/st_idt{}.csv'.format(i + 1) for i in range(1)]
        exp_comX, self.exp_TPX, exp_idt, exp_UF = load_st_idt(st_idt_paths, self.mech_path)

        files = glob.glob(os.path.join(ckpt_dir, 'sg_cond*.pt'))
        conds = sorted(int(re.search(r'sg_cond(\d+)\.pt', os.path.basename(f)).group(1)) for f in files)

        self.nets, y_mu_l, y_sd_l, y_obs_l, sigma_l = [], [], [], [], []
        ref_idx, red_labels, ndim = None, None, None
        for j in conds:
            ck_path = os.path.join(ckpt_dir, f'sg_cond{j}.pt')
            ckpt = torch.load(ck_path, map_location=self.device, weights_only=False)
            idx_j = np.asarray(ckpt['red_rxn_idx'])
            if ref_idx is None:
                ref_idx, red_labels, ndim = idx_j, list(ckpt['red_labels']), len(idx_j)
            elif not np.array_equal(idx_j, ref_idx):
                raise ValueError(f"工况 {j} 的 red_rxn_idx 与参考不一致, 无法共享 theta")
            self.nets.append(build_net(ckpt, self.device).eval())
            y_mu_l.append(float(ckpt['y_mu']))
            y_sd_l.append(float(ckpt['y_sd']))
            y_obs_l.append(float(np.log(exp_idt[j] / 1e3)))  # ms -> s
            sigma_l.append(np.log(float(exp_UF[j])))

        self.y_mu = torch.tensor(y_mu_l, dtype=torch.float32, device=self.device)
        self.y_sd = torch.tensor(y_sd_l, dtype=torch.float32, device=self.device)
        self.y_obs = torch.tensor(y_obs_l, dtype=torch.float32, device=self.device)
        self.sigma = torch.tensor(sigma_l, dtype=torch.float32, device=self.device)

        self.input_num = len(idx_j)
        self.output_num = len(self.nets)
        self.surrogate = Surrogate("highdim", lambda x: self.solve_t(self.transform(x)),
                                   self.input_num, self.output_num,
                                   torch.Tensor([[-3, 3]] * self.input_num), 20)

    @torch.no_grad()
    def solve_t(self, params):
        t = params if torch.is_tensor(params) else torch.tensor(params)
        t = t.to(self.device)
        outs = [self.nets[c](t).squeeze(-1) * self.y_sd[c] + self.y_mu[c]
                for c in range(self.output_num)]
        return torch.stack(outs, dim=1)

    def evalNegLL_t(self, modelOut):
        resid = (modelOut - self.y_obs) / self.sigma  # [batch, K] 广播
        ll3 = -0.5 * torch.sum(resid ** 2, dim=1, keepdim=True)  # [batch, 1]
        return -ll3

    def transform(self, x):
        return torch.tanh(x)  # 无约束隐空间 -> 盒 [-1,1]

    def den_t(self, x, surrogate=True):
        batch_size = x.size(0)
        adjust = torch.sum(np.log(4.0) + 2.0 * x - 2.0 * F.softplus(2.0 * x), dim=1, keepdim=True)
        if surrogate:
            modelOut = self.surrogate.forward(x)
        else:
            modelOut = self.solve_t(self.transform(x))
        return - self.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust


if __name__ == '__main__':
    model = Highdim()
    model.surrogate.surrogate_load()
    # model.surrogate.gen_grid(gridnum=20000, store=True)
    # model.surrogate.pre_train(200, 1e-3, 0.9999, 5, store=True)

    #theta = torch.tensor(np.load('./hmc_logs/N_mc100000/posterior_hmc.npy'), dtype=torch.double)

    # eps = 1e-6
    # theta = theta.clamp(-1 + eps, 1 - eps)  # 防止贴边 -> atanh 爆 inf
    # xt = torch.atanh(theta)  # [-1,1] -> ℝ 隐空间


    xt = torch.tensor(np.loadtxt('./results/samples20000'), dtype=torch.double)

    pred = model.surrogate.forward(xt)  # surrogate 期望 latent x
    true = model.solve_t(model.transform(xt))  # = solve_t(tanh(atanh(theta))) = solve_t(theta)

    print("MAE (log-IDT):", ((torch.exp(pred) - torch.exp(true)) / torch.exp(true)).abs().mean().item())

    N, d = np.load('./hmc_logs/N_mc100000/posterior_hmc.npy').shape
    g = torch.Generator().manual_seed(0)
    theta = torch.rand(N, d, generator=g, dtype=torch.double) * 2 - 1

    eps = 1e-6
    theta = theta.clamp(-1 + eps, 1 - eps)
    xt = torch.atanh(theta)
    # xt = torch.tensor(np.loadtxt('./results0/samples50000'), dtype=torch.double)

    pred = model.surrogate.forward(xt)  # surrogate 期望 latent x
    true = model.solve_t(model.transform(xt))  # = solve_t(tanh(atanh(theta))) = solve_t(theta)

    print("MAE (log-IDT):", ((torch.exp(pred) - torch.exp(true)) / torch.exp(true)).abs().mean().item())
