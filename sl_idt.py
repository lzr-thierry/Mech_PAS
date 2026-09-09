import os
import argparse
import warnings
from scipy.stats import qmc
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import load_st_idt, calc_st_idt, count_params, split_exp_group

warnings.filterwarnings("ignore")


def eval_lnidt(x, base_cond, red_rxn_idx):
    cond = dict(base_cond)
    cond['normA'] = np.zeros(len(base_cond['A_tab']))
    cond['normA'][red_rxn_idx] = x
    tau = calc_st_idt(cond)
    if not np.isfinite(tau) or tau <= 0.0:  # 非着火 / 窗口截断
        return np.nan
    return float(np.log(tau))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mech_path', type=str, default='./nh3_mech/NH3_Zhang2023_KAUST.yaml')
    parser.add_argument('--UF', type=float, default=10)
    # parser.add_argument('--cond', type=int, default=4, help='experimental condition index')
    # parser.add_argument('--N_mc', type=int, default=1000, help='Monte-Carlo samples for uncertainty propagation')
    parser.add_argument('--sens_thresh', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='./sl_logs')
    args = parser.parse_args()

    mech_path = args.mech_path
    st_idt_paths = ['./exp_data/nh3/st_idt{}.csv'.format(i + 1) for i in range(1)]

    exp_comX, exp_TPX, exp_idt, exp_UF = load_st_idt(st_idt_paths, args.mech_path)

    P_min = float(exp_TPX[:, 1].min())
    P_max = float(exp_TPX[:, 1].max())
    A_dim, A_tab = count_params(args.mech_path, P_min=P_min, P_max=P_max)
    # labels = label_params(args.mech_path, A_tab)

    red_rxn_idx = set()
    ds = pd.read_csv('./lsa_logs/rxn_sens_all.csv')

    group_indices = split_exp_group(exp_comX, exp_TPX)
    exp_idx = []
    for iii in range(len(group_indices) - 1):
        exp_idx += range(group_indices[iii], group_indices[iii + 1])
    print(len(exp_idx), exp_idx)
    for jj in exp_idx:
        dsi = ds['exp{}'.format(jj)]
        red_rxn_idx |= set(np.where(dsi >= 0.01)[0])
    print(len(red_rxn_idx))
    red_rxn_idx = sorted(red_rxn_idx)  # 固定顺序
    red_rxn_idx = np.array(red_rxn_idx)  # 方便花式索引

    red_labels = ds['rxn_label'].iloc[red_rxn_idx].tolist()
    red_A_dim = len(red_labels)

    for N_mc in [100000]:
        out_dir = args.out_dir + './N_mc{}'.format(N_mc)
        os.makedirs(out_dir, exist_ok=True)

        # j = args.cond
        if os.path.exists(out_dir + '/mc_X.npy'):
            X_sample = np.load(out_dir + '/mc_X.npy')
        else:
            X_sample = qmc.scale(qmc.LatinHypercube(d=red_A_dim).random(n=N_mc), -1, 1)
            np.save(out_dir + '/mc_X.npy', X_sample)

        for j in exp_idx:
            base_cond = {'mech_path': args.mech_path,
                         'A_tab':     A_tab,
                         'param_UF':  np.full(A_dim, args.UF),
                         'TPX':       [float(exp_TPX[j, 0]), float(exp_TPX[j, 1]), float(exp_TPX[j, 2])],
                         'comX':      exp_comX[j],
                         'idt':       float(exp_idt[j])}

            # ---- compute or load MC samples -------------------------------------
            npy_path = os.path.join(out_dir, f'mc_cond{j}')

            if os.path.exists(npy_path + '_Y.npy'):
                print(f"loading cached {npy_path}")
                Y_sample = np.load(npy_path + '_Y.npy')
            else:
                Y_sample = np.asarray(Parallel(n_jobs=-1, backend="loky", verbose=5)(
                    delayed(eval_lnidt)(X_sample[k], base_cond, red_rxn_idx) for k in range(X_sample.shape[0])))
                np.save(npy_path + '_Y.npy', Y_sample)
                print(f"wrote {npy_path}")


            # ---- drop non-igniting samples --------------------------------------
            mask = np.isfinite(Y_sample)
            n_drop = int((~mask).sum())
            if n_drop:
                print(f"{n_drop}/{len(mask)} non-igniting samples dropped")
            X_sample, Y_sample = X_sample[mask], Y_sample[mask]

            # ---- train / val / test split 7:2:1 --------------------------------
            torch.manual_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            N = X_sample.shape[0]
            perm = rng.permutation(N)
            n_tr, n_va = int(0.7 * N), int(0.2 * N)
            idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

            # standardize using TRAIN stats only
            y_mu, y_sd = Y_sample[idx_tr].mean(), Y_sample[idx_tr].std() + 1e-8
            Yn = (Y_sample - y_mu) / y_sd

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Xt = torch.tensor(X_sample, dtype=torch.float32, device=device)
            Yt = torch.tensor(Yn, dtype=torch.float32, device=device).unsqueeze(1)

            tr_loader = DataLoader(TensorDataset(Xt[idx_tr], Yt[idx_tr]),
                                   batch_size=args.batch_size, shuffle=True)

            # ---- surrogate: 2 hidden layers, tanh, 64 neurons -------------------
            net = nn.Sequential(
                nn.Linear(red_A_dim, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 1),
            ).to(device)

            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            loss_fn = nn.MSELoss()

            # ---- training loop --------------------------------------------------
            best_va, best_state = float('inf'), None
            history = []
            for ep in range(args.epochs):
                net.train()
                tr_loss_sum, n_b = 0.0, 0
                for xb, yb in tr_loader:
                    opt.zero_grad()
                    loss = loss_fn(net(xb), yb)
                    loss.backward()
                    opt.step()
                    tr_loss_sum += loss.item()
                    n_b += 1
                tr_loss = tr_loss_sum / n_b

                net.eval()
                with torch.no_grad():
                    va_loss = loss_fn(net(Xt[idx_va]), Yt[idx_va]).item()
                history.append((ep + 1, tr_loss, va_loss))

                if va_loss < best_va:  # keep best-on-val
                    best_va = va_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                if (ep + 1) % 20 == 0:
                    print(f"epoch {ep + 1:4d}  train_mse={tr_loss:.4e}  val_mse={va_loss:.4e}")

            net.load_state_dict(best_state)

            # ---- test metrics (back in ln(tau) units) ---------------------------
            net.eval()
            with torch.no_grad():
                pred_te = net(Xt[idx_te]).cpu().numpy().ravel() * y_sd + y_mu
            true_te = Y_sample[idx_te]
            test_r2 = r2_score(true_te, pred_te)
            test_mre = np.mean(np.abs(np.exp(pred_te) - np.exp(true_te)) / np.exp(true_te))
            print(f"\ntest R²   = {test_r2:.4f}")
            print(f"test RE = {test_mre:.4e}")

            # ---- save training history (train & val loss) -----------------------
            hist_df = pd.DataFrame(history, columns=['epoch', 'train_mse_std', 'val_mse_std'])
            hist_csv = os.path.join(out_dir, f'train_history_cond{j}.csv')
            hist_df.to_csv(hist_csv, index=False)
            print(f"wrote {hist_csv}")

            # ---- save final network weights + reuse metadata --------------------
            ckpt = os.path.join(out_dir, f'sg_cond{j}.pt')
            torch.save({
                'state_dict': best_state,
                'red_rxn_idx': red_rxn_idx,
                'red_labels': red_labels,
                'y_mu': float(y_mu), 'y_sd': float(y_sd),
            }, ckpt)
            print(f"wrote {ckpt}  (test R²={test_r2:.4f}, MRE={test_mre:.4e})")


