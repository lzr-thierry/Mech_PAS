import os
import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import argparse
from utils import load_st_idt, calc_st_idt, count_params, label_params, split_exp_group
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mech_path', type=str, default='./nh3_mech/NH3_Zhang2023_KAUST.yaml')
    parser.add_argument('--UF', type=float, default=1.1)
    parser.add_argument('--out_dir', type=str, default='./lsa_logs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mech_path = args.mech_path
    st_idt_paths = ['./exp_data/nh3/st_idt{}.csv'.format(i + 1) for i in range(1)]

    exp_comX, exp_TPX, exp_idt, exp_UF = load_st_idt(st_idt_paths, args.mech_path)
    n_exp = len(exp_comX)
    P_min = float(exp_TPX[:, 1].min())
    P_max = float(exp_TPX[:, 1].max())

    A_dim, A_tab = count_params(args.mech_path, P_min=P_min, P_max=P_max)
    param_UF = np.full(A_dim, args.UF)
    print(f"tunable parameters : {A_dim}")

    idt0_file = os.path.join(args.out_dir, 'idt0.npy')
    idts_file = os.path.join(args.out_dir, 'idts.npy')

    group_indices = split_exp_group(exp_comX, exp_TPX)

    print("loading cached idt0.npy / idts.npy ...")
    idt0 = np.load(idt0_file)
    idts = np.load(idts_file)

    # ---- sensitivity coefficients -------------------------------------------
    lnUF = np.log(param_UF)[:, None]  # (A_dim, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        # S = np.log(idts / idt0[None, :]) / lnUF  # signed, normalised
        S = np.abs(idts - idt0[None, :]) / idt0[None, :]  # original metric
        rel = (idts - idt0[None, :]) / idt0[None, :]  # original metric

    labels = label_params(args.mech_path, A_tab)
    exp_cols = [f"exp{j}" for j in range(n_exp)]

    ds = pd.DataFrame(S, index=labels, columns=exp_cols)
    ds["mean_abs_sens"] = np.abs(S).mean(axis=1)
    ds["max_abs_sens"] = np.abs(S).max(axis=1)
    ds["mean_rel_change"] = rel.mean(axis=1)
    ds = ds.sort_values(by="mean_abs_sens", ascending=False)

    ds.index.name = "rxn_label"
    all_csv = os.path.join(args.out_dir, "rxn_sign_sens_all.csv")
    ds.to_csv(all_csv)
    print(f"wrote {all_csv}")

    # # ---- export the top-N ranked reactions as a parameter table --------------
    # label_to_entry = {lab: entry for lab, entry in zip(labels, A_tab)}
    # top_labels = ds.index.tolist()[:args.top_n]
    # sorted_table = [label_to_entry[lab] for lab in top_labels]
    #
    # table_json = os.path.join(args.out_dir, "table_sorted.json")
    # with open(table_json, 'w') as f:
    #     json.dump(sorted_table, f, indent=2)
    # print(f"wrote {table_json}  (top {args.top_n})")
    #
    # print("\ntop 15 most sensitive parameters:")
    # print(ds[["mean_abs_sens", "max_abs_sens"]].head(15).to_string())