import numpy as np
import pandas as pd
import cantera as ct
import os
import torch


# Species we recognise in the experimental files.
FUEL_KEYS = ['NH3', 'H2']
OXIDIZER_KEYS = ['O2', 'N2', 'AR']


def load_st_idt(data_paths, mech_path):
    rows = []
    for data_file in data_paths:
        rows += pd.read_csv(data_file).to_dict('records')

    gas = ct.Solution(mech_path)

    exp_comX, exp_TPX, exp_idt, exp_UF = [], [], [], []

    for row in rows:
        comX = {'fuel':{k: float(row[k]) for k in FUEL_KEYS if k in row},
                'oxidizer':{k: float(row[k]) for k in OXIDIZER_KEYS if k in row}}

        gas.TP = float(row['T']), float(row['P']) * ct.one_atm
        gas.X = comX['fuel'] | comX['oxidizer']
        phi = gas.equivalence_ratio()  # elemental definition, Ar ignored

        exp_comX.append(comX)
        exp_TPX.append([float(row['T']), float(row['P']), phi])
        exp_idt.append(float(row['t']))
        exp_UF.append(float(row['UF']))

    return exp_comX, np.array(exp_TPX), np.array(exp_idt), np.array(exp_UF)


def modify_A(gas, normA, table, UF):
    for i, entry in enumerate(table):
        rxn_idx   = entry['rxn_idx']
        rate_type = entry['rate_type']
        local     = entry['local']
        rxn = gas.reactions()[rxn_idx]

        if rate_type == 'arrhenius':
            r = rxn.rate
            A = r.pre_exponential_factor * UF[i] ** normA[i]
            rxn.rate = ct.ArrheniusRate(A, r.temperature_exponent, r.activation_energy)
        elif rate_type == 'plog':
            new_rates = list(rxn.rate.rates)
            P, arr = new_rates[local]
            A = arr.pre_exponential_factor * UF[i] ** normA[i]
            new_rates[local] = (P, ct.Arrhenius(A, arr.temperature_exponent, arr.activation_energy))
            rxn.rate = ct.PlogRate(new_rates)
        else:  # falloff
            r = rxn.rate
            if local == 'low':
                A = r.low_rate.pre_exponential_factor * UF[i] ** normA[i]
                low  = ct.Arrhenius(A, r.low_rate.temperature_exponent, r.low_rate.activation_energy)
                high = ct.Arrhenius(r.high_rate.pre_exponential_factor, r.high_rate.temperature_exponent,
                                    r.high_rate.activation_energy)
            else:  # 'high'
                A = r.high_rate.pre_exponential_factor * UF[i] ** normA[i]
                low  = ct.Arrhenius(r.low_rate.pre_exponential_factor, r.low_rate.temperature_exponent,
                                    r.low_rate.activation_energy)
                high = ct.Arrhenius(A, r.high_rate.temperature_exponent, r.high_rate.activation_energy)

            if type(r) == ct.LindemannRate:
                rxn.rate = ct.LindemannRate(low=low, high=high, falloff_coeffs=r.falloff_coeffs)
            else:
                rxn.rate = ct.TroeRate(low=low, high=high, falloff_coeffs=r.falloff_coeffs)

        gas.modify_reaction(rxn_idx, rxn)

    return gas


def calc_st_idt(condition, max_factor=100, dT_floor=50.0):
    mech_path  = condition['mech_path']
    normA      = condition['normA']
    A_tab      = condition['A_tab']
    param_UF   = condition['param_UF']
    T, P, phi  = condition['TPX']
    comX       = condition['comX']
    idt        = condition['idt']

    gas = ct.Solution(mech_path)
    gas = modify_A(gas, normA, A_tab, param_UF)
    gas.set_equivalence_ratio(phi, fuel=comX['fuel'], oxidizer=comX['oxidizer'], basis='mole')
    gas.TP = T, ct.one_atm * P

    r   = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim.rtol = 1e-8
    sim.atol = 1e-15

    t_end = idt / 1000 * max_factor
    time, temp = [], []
    while sim.time < t_end:
        sim.step()
        time.append(sim.time)
        temp.append(r.T)                     # 当前反应器温度

    time = np.array(time)
    temp = np.array(temp)

    dTdt = np.diff(temp) / np.diff(time)
    tmid = 0.5 * (time[1:] + time[:-1])
    k = int(np.argmax(dTdt))

    if 0 < k < len(dTdt) - 1:                 # 抛物线顶点 → 亚网格精度
        y0, y1, y2 = dTdt[k - 1], dTdt[k], dTdt[k + 1]
        den = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
        return tmid[k] + delta * 0.5 * (tmid[k + 1] - tmid[k - 1])
    return tmid[k]


def count_params(mech_path, P_min, P_max, rtol=1e-6):
    gas = ct.Solution(mech_path)
    lo, hi = P_min * (1.0 - rtol) * ct.one_atm, P_max * (1.0 + rtol) * ct.one_atm
    table = []

    for rxn_idx, rxn in enumerate(gas.reactions()):
        rate = rxn.rate

        if type(rate) == ct.ArrheniusRate:
            table.append({'rxn_idx': rxn_idx,
                          'rate_type': 'arrhenius',
                          'local': 'A'})
        elif type(rate) == ct.PlogRate:
            pressures = [P for P, _ in rate.rates]
            i_lo = 0
            for j, P in enumerate(pressures):
                if P <= lo:
                    i_lo = j
            i_hi = len(pressures) - 1
            for j, P in enumerate(pressures):
                if P >= hi:
                    i_hi = j
                    break
            for j in range(i_lo, i_hi + 1):
                table.append({'rxn_idx': rxn_idx,
                              'rate_type': 'plog',
                              'local': j})
        else:  # LindemannRate / TroeRate
            table.append({'rxn_idx': rxn_idx,
                          'rate_type': 'falloff',
                          'local': 'low'})
            table.append({'rxn_idx': rxn_idx,
                          'rate_type': 'falloff',
                          'local': 'high'})
    return len(table), table


def label_params(mech_path, table):
    """Human-readable label for each tabulated parameter (for csv index)."""
    gas = ct.Solution(mech_path)
    labels = []
    for entry in table:
        rxn_idx = entry['rxn_idx']
        eq = gas.reactions()[rxn_idx].equation
        if entry['rate_type'] == 'arrhenius':
            labels.append(f"r{rxn_idx} {eq}")
        elif entry['rate_type'] == 'plog':
            P_atm = gas.reactions()[rxn_idx].rate.rates[entry['local']][0] / ct.one_atm
            labels.append(f"r{rxn_idx} plog[{entry['local']}]@{P_atm:.1f}atm {eq}")
        else:
            labels.append(f"r{rxn_idx} {entry['local']} {eq}")
    return labels


def split_exp_group(exp_comX, exp_TPX, p_rtol=1):
    n = len(exp_comX)
    bounds = [0]
    X_ref = exp_comX[0]
    P_ref = exp_TPX[0, 1]
    for i in range(1, n):
        X = exp_comX[i]
        P = exp_TPX[i, 1]
        if X != X_ref or abs(P - P_ref) > p_rtol * P_ref:
            bounds.append(i)
            X_ref, P_ref = X, P
    bounds.append(n)
    return bounds
