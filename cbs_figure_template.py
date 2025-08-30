
"""
CBS Figure Template (Owens/APA style)
------------------------------------
This module generates Owens-style growth figures with 95% CI ribbons from
Mplus parameter estimates and TECH3 covariance matrices.

Features
- Optional moderator (±1 SD) with line coding: solid = high, dashed = low
- Optional quadratic time term (t^2)
- Grayscale APA-friendly styling (Enhanced = black, Standard = gray)
- Markers at each timepoint (squares = Enhanced, circles = Standard)
- Exports an Excel file with Predicted, Lower CI, Upper CI
- Exports a PNG figure

Time coding
- Assumes BW6 centering by default: Timepoint 1..6 mapped to t = -5..0.
- You can override by passing a list of integer time scores (e.g., [-5,-4,-3,-2,-1,0]).

Inputs
- params: pandas Series or dict with the parameters you are using.
          Supported keys (use only those you need):
            'alpha_int'    : Intercept mean (Between)
            'alpha_slope'  : Linear slope mean (Between)
            'b_int_cond'   : Condition -> intercept
            'b_int_mod'    : Moderator -> intercept
            'b_int_intx'   : Cond x Mod -> intercept
            'b_slope_cond' : Condition -> slope
            'b_slope_mod'  : Moderator -> slope
            'b_slope_intx' : Cond x Mod -> slope
            'quad'         : Quadratic coefficient on t^2 (Within; fixed effect)

- Sigma: pandas DataFrame (square) where both rows and columns are the *parameter keys*
         included above and the entries are covariances from TECH3. The order does not
         matter as long as row/column names match param keys.

Flags
- include_moderator: bool (default True)
- include_quadratic: bool (default False)
- moderator_sd: float (default 1.0)
- time_scores: list[int] length K (default [-5,-4,-3,-2,-1,0] mapping Timepoint 1..6)
- y_label: str (axis label)
- output_excel_path: str
- output_png_path: str

Example
-------
from cbs_figure_template import generate_cbs_figure

params = {
    'alpha_int': 59.576, 'alpha_slope': -3.550,
    'b_int_cond': -6.304, 'b_int_mod': 6.357, 'b_int_intx': 7.906,
    'b_slope_cond': -1.380, 'b_slope_mod': 1.485, 'b_slope_intx': 2.496,
    'quad': -1.113
}

Sigma = pd.read_csv('example_arrv_sigma.csv', index_col=0)  # rows/cols named by keys above

generate_cbs_figure(params=params, Sigma=Sigma,
                    include_moderator=True, include_quadratic=True,
                    y_label='Percent Teacher Responding Appropriately',
                    output_excel_path='ARRV_template_output.xlsx',
                    output_png_path='ARRV_template_figure.png')
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class FigureOptions:
    include_moderator: bool = True
    include_quadratic: bool = False
    moderator_sd: float = 1.0
    time_scores: Optional[List[int]] = None   # e.g., [-5,-4,-3,-2,-1,0]
    y_label: str = 'Outcome'
    x_label: str = 'Observation Timepoint (1–6)'
    output_excel_path: str = 'predictions_with_ci.xlsx'
    output_png_path: str = 'figure.png'


def _build_design_row(t:int, Cond:int, Imp:float,
                      include_moderator:bool, include_quadratic:bool) -> Tuple[List[str], np.ndarray]:
    """
    Returns (param_keys_in_order, derivative vector X) for Var(Yhat) = X' Sigma X.
    Only includes derivatives for parameters present in the keys list.
    """
    # Mapping from conceptual parameter to derivative component
    derivs = {
        'alpha_slope': t,
        'alpha_int': 1.0,
        'b_slope_mod': (Imp * t),
        'b_slope_intx': (Cond * Imp * t),
        'b_slope_cond': (Cond * t),
        'b_int_mod': Imp,
        'b_int_intx': (Cond * Imp),
        'b_int_cond': Cond,
        'quad': (t**2),
    }
    # Prune derivatives according to flags
    keys = ['alpha_slope', 'alpha_int']
    if include_moderator:
        keys += ['b_slope_mod', 'b_slope_intx', 'b_slope_cond', 'b_int_mod', 'b_int_intx', 'b_int_cond']
    else:
        keys += ['b_slope_cond', 'b_int_cond']
    if include_quadratic: keys += ['quad']
    X = np.array([derivs[k] for k in keys], dtype=float)
    return keys, X


def _predict_mean(t:int, Cond:int, Imp:float, p:Dict[str,float],
                  include_moderator:bool, include_quadratic:bool) -> float:
    # Base
    val = p.get('alpha_int', 0.0) + p.get('alpha_slope', 0.0)*t
    # Condition, moderator, interactions
    if include_moderator:
        val += (p.get('b_int_cond', 0.0)*Cond +
                p.get('b_int_mod', 0.0)*Imp +
                p.get('b_int_intx', 0.0)*Cond*Imp)
        val += (p.get('b_slope_cond', 0.0)*Cond +
                p.get('b_slope_mod', 0.0)*Imp +
                p.get('b_slope_intx', 0.0)*Cond*Imp) * t
    else:
        val += (p.get('b_int_cond', 0.0)*Cond)
        val += (p.get('b_slope_cond', 0.0)*Cond) * t
    # Quadratic
    if include_quadratic:
        val += p.get('quad', 0.0)*(t**2)
    return val


def generate_cbs_figure(params:Dict[str,float],
                        Sigma:pd.DataFrame,
                        include_moderator:bool=True,
                        include_quadratic:bool=False,
                        moderator_sd:float=1.0,
                        time_scores:Optional[List[int]]=None,
                        y_label:str='Outcome',
                        x_label:str='Observation Timepoint (1–6)',
                        output_excel_path:str='predictions_with_ci.xlsx',
                        output_png_path:str='figure.png'
                        ) -> pd.DataFrame:
    """
    Generate predictions + 95% CIs and an APA/OWENS style figure.
    Returns the unrounded DataFrame of predictions.
    """
    # Time coding
    if time_scores is None:
        time_scores = [-5,-4,-3,-2,-1,0]
    # Build covariance matrix restricted to used keys
    # We will create Sigma_sub in the order of the derivative keys used.
    conditions = ['Standard','Enhanced']
    moderators = ['Low','High'] if include_moderator else ['Average']
    rows = []
    for tp_idx, t in enumerate(time_scores, start=1):
        for cond_label in conditions:
            Cond = 1 if cond_label=='Enhanced' else 0
            for mod_label in moderators:
                if include_moderator:
                    Imp = +moderator_sd if mod_label=='High' else -moderator_sd
                else:
                    Imp = 0.0
                # Prediction
                yhat = _predict_mean(t, Cond, Imp, params, include_moderator, include_quadratic)
                # Derivatives / design row
                keys, X = _build_design_row(t, Cond, Imp, include_moderator, include_quadratic)
                # Create Sigma_sub (order by keys)
                # Check Sigma has all required keys
                missing = [k for k in keys if k not in Sigma.index or k not in Sigma.columns]
                if len(missing)>0:
                    raise ValueError(f"Sigma is missing covariance rows/cols for: {missing}")
                Sigma_sub = Sigma.loc[keys, keys].to_numpy(dtype=float)
                var = float(X @ Sigma_sub @ X)
                se = np.sqrt(max(var, 0.0))
                lci, uci = yhat - 1.96*se, yhat + 1.96*se
                rows.append({
                    'Timepoint (1–6)': tp_idx,
                    'Condition': cond_label,
                    'Moderator': mod_label,
                    'Predicted Value': yhat,
                    'Lower CI': lci,
                    'Upper CI': uci
                })
    df = pd.DataFrame(rows)
    # Save Excel (rounded to 2 decimals to match manuscript reporting)
    df_out = df.copy()
    for c in ['Predicted Value','Lower CI','Upper CI']:
        df_out[c] = df_out[c].round(2)
    df_out.to_excel(output_excel_path, index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    # Aesthetics per Owens template
    specs = []
    if include_moderator:
        specs = [
            ('Standard','Low','gray','--','o'),
            ('Standard','High','gray','-','o'),
            ('Enhanced','Low','black','--','s'),
            ('Enhanced','High','black','-','s'),
        ]
    else:
        specs = [
            ('Standard','Average','gray','--','o'),
            ('Enhanced','Average','black','-','s'),
        ]
    for cond, mod, color, ls, marker in specs:
        sub = df[(df['Condition']==cond) & (df['Moderator']==mod)].sort_values('Timepoint (1–6)')
        x = sub['Timepoint (1–6)'].to_numpy()
        y = sub['Predicted Value'].to_numpy()
        l = sub['Lower CI'].to_numpy()
        u = sub['Upper CI'].to_numpy()
        ax.fill_between(x, l, u, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2, zorder=3)
        ax.scatter(x, y, color=color, marker=marker, s=40, edgecolor=color, zorder=4)
    # Labels, legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, marker='o', markersize=7, label='Standard, Low' if include_moderator else 'Standard'),
        mlines.Line2D([], [], color='gray', linestyle='-',  linewidth=2, marker='o', markersize=7, label='Standard, High' if include_moderator else '—'),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, marker='s', markersize=7, label='Enhanced, Low' if include_moderator else 'Enhanced'),
        mlines.Line2D([], [], color='black', linestyle='-',  linewidth=2, marker='s', markersize=7, label='Enhanced, High' if include_moderator else '—'),
    ]
    # Remove empty labels when no moderator
    legend_handles = [h for h in legend_handles if h.get_label() not in ['—']]
    ax.legend(legend_handles, [h.get_label() for h in legend_handles],
              frameon=False, handlelength=4, handletextpad=1)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    return df
    

# ---------------------------
# Example usage (ARRV demo)
# ---------------------------
if __name__ == '__main__':
    # Example parameters (ARRV)
    params = {
        'alpha_int': 59.576, 'alpha_slope': -3.550,
        'b_int_cond': -6.304, 'b_int_mod': 6.357, 'b_int_intx': 7.906,
        'b_slope_cond': -1.380, 'b_slope_mod': 1.485, 'b_slope_intx': 2.496,
        'quad': -1.113
    }
    # Example covariance: replace with the TECH3 covariances for your run.
    # IMPORTANT: Rows/cols must be named to match the parameter keys above.
    Sigma = pd.read_csv('example_arrv_sigma.csv', index_col=0)
    generate_cbs_figure(params=params, Sigma=Sigma,
                        include_moderator=True, include_quadratic=True,
                        y_label='Percent Teacher Responding Appropriately',
                        output_excel_path='ARRV_template_output.xlsx',
                        output_png_path='ARRV_template_figure.png')
