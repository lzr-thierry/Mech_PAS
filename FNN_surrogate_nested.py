import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import qmc
from os import path

torch.set_default_tensor_type(torch.DoubleTensor)


class FNN1D(nn.Module):
    """单个 1D 输出子网络: input_size -> 64 -> 64 -> 1"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)          # 最后一层不加激活, 与原代码一致


class FNN(nn.Module):
    """output_size 个相互独立的子网络的组合; forward 返回 (batch, output_size)"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.nets = nn.ModuleList([FNN1D(input_size) for _ in range(output_size)])

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=-1)


class Surrogate:
    def __init__(self, model_name, model_func, input_size, output_size, limits=None, memory_len=20):
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.mf = model_func
        self.pre_out = None
        self.m = None
        self.sd = None
        self.tm = None
        self.tsd = None
        self.limits = limits
        self.pre_grid = None
        self.surrogate = FNN(input_size, output_size)
        self.beta_0 = 1
        self.beta_1 = 0.1

        self.memory_grid = []
        self.memory_out = []
        self.memory_len = memory_len
        self.weights = torch.Tensor([np.exp(-self.beta_1 * i) for i in range(memory_len)])
        self.grid_record = None

    @property
    def limits(self):
        return self.__limits

    @limits.setter
    def limits(self, limits):
        limits = torch.Tensor(limits).tolist()
        if len(limits) != self.input_size:
            print("Error: Invalid input size for limit. Abort.")
            exit(-1)
        elif any(len(item) != 2 for item in limits):
            print("Error: Limits should be two bounds. Abort.")
            exit(-1)
        elif any(item[0] > item[1] for item in limits):
            print("Error: Upper bound should not be smaller than lower bound. Abort.")
            exit(-1)
        self.__limits = limits

    @property
    def pre_grid(self):
        return self.__pre_grid

    @pre_grid.setter
    def pre_grid(self, pre_grid):
        if pre_grid is None:
            if path.exists(self.model_name + '.npz'):
                container = np.load(self.model_name + '.npz')
                if 'pre_grid' in container:
                    self.pre_grid = container['pre_grid']
                    print(self.pre_grid.size())
                    print("Success: Pre-Grid found.")
            else:
                print("Warning: " + self.model_name + ".npz does not found, please generate pre-grid.")
                print("Suggestion: Use Surrogate.gen_grid(input_limits=None, grid_num=5, store=True)")
        else:
            self.__pre_grid = torch.Tensor(pre_grid)
            self.m = torch.mean(self.pre_grid, 0)
            self.sd = torch.std(self.pre_grid, 0)
            ys = []
            for i in [20, 21, 22, 23, 24, 25, 26, 27,
                      47, 48, 49, 50, 51, 52, 53, 54, 55,
                      97, 98, 99, 100, 101, 102, 103]:
                ys.append(np.load('./sl_logs/N_mc{}/mc_cond{}_Y.npy'.format(20000, i)))

            self.pre_out = torch.tensor(np.stack(ys, axis=1))
            self.tm = torch.mean(self.pre_out, 0)
            self.tsd = torch.std(self.pre_out, 0)
            self.grid_record = self.__pre_grid.clone()

    def gen_grid(self, gridnum=1000, store=True, theta_max=0.99999):
        theta = np.load('./sl_logs/N_mc{}/mc_X.npy'.format(gridnum)).clip(-theta_max, theta_max)
        x = np.arctanh(theta)  # 反映射回 latent x
        grid = torch.tensor(x)
        if store:
            self.pre_grid = grid  # pre_out = mf(x) = solve_t(tanh(x)) = solve_t(θ)
            self.grid_record = self.pre_grid.clone()
            self.surrogate_save(0)
        return grid

    def surrogate_save(self, iter):
        torch.save(self.surrogate.state_dict(), self.model_name + '_{}.sur'.format(iter))
        np.savez(self.model_name, limits=self.limits, pre_grid=self.pre_grid,
                 grid_record=self.grid_record)

    def surrogate_load(self):
        self.surrogate.load_state_dict(torch.load(self.model_name + '_88000.sur'))
        container = np.load(self.model_name + '.npz')
        for key in container:
            try:
                setattr(self, key, torch.Tensor(container[key]))
                print("Success: [" + key + "] loaded.")
            except:
                print("Warning: [" + key + "] is not a surrogate variables.")

    def pre_train(self, max_iters, lr, lr_exp, record_interval, store=True, reg=False, batch_size=128):
        grid = (self.pre_grid - self.m) / self.sd
        out = (self.pre_out - self.tm) / self.tsd
        N = grid.size(0)

        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)

        self.surrogate.train()
        for i in range(max_iters):  # 一个 i = 一个 epoch
            perm = torch.randperm(N)
            loss_sum, n_b = 0.0, 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                y = self.surrogate(grid[idx])
                loss = torch.mean((y - out[idx]) ** 2)
                if reg:
                    reg_loss = sum(p.abs().sum() for p in self.surrogate.parameters()) * 0.0001
                    loss = loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                n_b += 1
            scheduler.step()  # 每个 epoch 衰减一次, 且放在更新之后

            if i % record_interval == 0:
                print('epoch {}   loss {:.4e}'.format(i, loss_sum / n_b))
        if store:
            self.surrogate_save(0)

    def update(self, x, max_iters=200, lr=0.001, lr_exp=0.9999, record_interval=500, store=False, tol=1e-5, reg=False, batch_size=128):
        self.grid_record = torch.cat((self.grid_record, x), dim=0)
        s = torch.std(x, dim=0)
        thresh = 0.1
        if torch.any(s < thresh):
            p = x[:, s < thresh]
            x[:, s < thresh] += torch.normal(0, 1, size=tuple(p.size())) * thresh
        print("Std: ", s)
        print("Std after: ", torch.std(x, dim=0))
        if len(self.memory_grid) >= self.memory_len:
            self.memory_grid.pop()
            self.memory_out.pop()
        self.memory_grid.insert(0, (x - self.m) / self.sd)
        self.memory_out.insert(0, (self.mf(x) - self.tm) / self.tsd)

        if self.beta_0 == 0:
            grid_all = torch.cat(self.memory_grid, dim=0)
            out_all = torch.cat(self.memory_out, dim=0)
        else:
            grid_all = torch.cat(((self.pre_grid - self.m) / self.sd, *self.memory_grid), dim=0)
            out_all = torch.cat(((self.pre_out - self.tm) / self.tsd, *self.memory_out), dim=0)

        N = grid_all.size(0)
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_exp)

        self.surrogate.train()
        for i in range(max_iters):  # 一个 i = 一个 epoch
            perm = torch.randperm(N)
            loss_sum, n_b = 0.0, 0
            for st in range(0, N, batch_size):
                idx = perm[st:st + batch_size]
                yb = self.surrogate(grid_all[idx])
                se = torch.mean((yb - out_all[idx]) ** 2, dim=1)  # 每样本 MSE(对 output 维平均)

                loss = se.mean()  # 整个 batch 一视同仁,严格等权

                if reg:
                    loss = loss + sum(p.abs().sum() for p in self.surrogate.parameters()) * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                n_b += 1
            scheduler.step()
            last = loss_sum / n_b
            print(i, 'loss:', last)
            if i % record_interval == 0:
                print('Updating: {}\t loss {:.4e}'.format(i, last), end='\r')
            if last < tol ** 2:
                break

        print(' ' * 60, end='\r')

    def forward(self, x):
        return self.surrogate((x - self.m) / self.sd) * self.tsd + self.tm


# if __name__ == '__main__':
#     S = Surrogate("Highdim", None, 3, 2, [[-7, 7], [-7, 7], [-7, 7]], 10, "Trivial_pregrid.txt")
#     print(S.pre_grid)
