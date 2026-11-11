"""Reproducible educational A/B test for an electronic service."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, ttest_ind

ROOT = Path(__file__).resolve().parent
DATA = ROOT /  'dataset.csv'
CHARTS = ROOT / 'outputs' / 'charts'
RESULTS = ROOT / 'outputs' / 'results'
for directory in (DATA.parent, CHARTS, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)
SEED = 20261111
ALPHA = 0.05
POWER = 0.80
BASELINE = 0.60
MDE_ABSOLUTE = 0.04
DAILY_USERS = 1600
GUARDRAIL_ERROR_MAX_INCREASE = 0.01
GUARDRAIL_TIME_MAX_INCREASE = 10.0


def sample_size_per_arm(p1, p2, alpha=0.05, power=0.8):
    """Normal approximation for two independent proportions, equal allocation."""
    z_alpha = norm.ppf(1-alpha/2)
    z_power = norm.ppf(power)
    pbar = (p1+p2)/2
    numerator = (z_alpha*math.sqrt(2*pbar*(1-pbar)) + z_power*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2
    return math.ceil(numerator / (p2-p1)**2)


def generate_data(n_per_arm):
    """Simulated independent users, randomized at user level, fixed seed."""
    rng = np.random.default_rng(SEED)
    n = 2*n_per_arm
    groups = np.array(['A']*n_per_arm + ['B']*n_per_arm)
    rng.shuffle(groups)
    completed = rng.binomial(1, np.where(groups == 'A', .60, .65))
    error = rng.binomial(1, np.where(groups == 'A', .025, .027))
    duration = np.maximum(10, rng.normal(np.where(groups == 'A', 190, 155), 45, n))
    df = pd.DataFrame({'user_id': [f'U{i:06d}' for i in range(1,n+1)],
                       'group': groups, 'completed': completed, 'error': error,
                       'duration_seconds': np.round(duration, 2)})
    df.to_csv(DATA, index=False)
    return df


def two_proportion_test(success_a, n_a, success_b, n_b):
    pa, pb = success_a/n_a, success_b/n_b
    effect = pb-pa
    pooled = (success_a+success_b)/(n_a+n_b)
    se_null = math.sqrt(pooled*(1-pooled)*(1/n_a+1/n_b))
    z = effect/se_null if se_null else 0
    p = 2*norm.sf(abs(z)) if se_null else 1
    se_ci = math.sqrt(pa*(1-pa)/n_a + pb*(1-pb)/n_b)
    ci = (effect - norm.ppf(.975)*se_ci, effect + norm.ppf(.975)*se_ci)
    return {'rate_a': pa, 'rate_b': pb, 'absolute_effect': effect,
            'relative_lift': effect/pa if pa else None, 'z': z,
            'p_value': p, 'ci95_lower': ci[0], 'ci95_upper': ci[1]}


def analyze(df, required_n):
    if df.isna().any().any():
        raise ValueError('Missing values detected: investigate before analysis.')
    if df.user_id.duplicated().any():
        raise ValueError('Duplicate users detected: investigate before analysis.')
    if set(df.group.unique()) != {'A','B'}:
        raise ValueError('Both groups A and B must be present.')
    if not df[['completed','error']].isin([0,1]).all().all():
        raise ValueError('Binary metrics contain invalid values.')
    if (df.duration_seconds <= 0).any():
        raise ValueError('Invalid duration detected.')
    a = df[df.group == 'A']
    b = df[df.group == 'B']
    na, nb = len(a), len(b)
    # SRM chi-square goodness-of-fit: expected equal split; threshold 0.001.
    expected = (na+nb)/2
    srm_chi2 = ((na-expected)**2+(nb-expected)**2)/expected
    srm_p = chi2.sf(srm_chi2, 1)
    primary = two_proportion_test(int(a.completed.sum()),na,int(b.completed.sum()),nb)
    errors = two_proportion_test(int(a.error.sum()),na,int(b.error.sum()),nb)
    ta, tb = float(a.duration_seconds.mean()), float(b.duration_seconds.mean())
    time_p = float(ttest_ind(a.duration_seconds,b.duration_seconds,equal_var=False).pvalue)
    srm_ok = srm_p >= .001
    sample_ok = min(na,nb) >= required_n
    error_ok = errors['ci95_upper'] <= GUARDRAIL_ERROR_MAX_INCREASE
    # Upper confidence bound for B-A time, Welch approximation (large samples).
    time_effect = tb-ta
    time_se = math.sqrt(a.duration_seconds.var(ddof=1)/na+b.duration_seconds.var(ddof=1)/nb)
    time_upper = time_effect + norm.ppf(.975)*time_se
    time_ok = time_upper <= GUARDRAIL_TIME_MAX_INCREASE
    if not srm_ok or not sample_ok:
        decision = 'DO NOT LAUNCH: experiment quality or sample-size check failed'
    elif primary['p_value'] < ALPHA and primary['absolute_effect'] > 0 and error_ok and time_ok:
        decision = 'LAUNCH: statistically significant improvement and guardrails passed'
    elif not error_ok or not time_ok:
        decision = 'DO NOT LAUNCH: guardrail safety threshold not established'
    else:
        decision = 'INCONCLUSIVE / DO NOT LAUNCH: insufficient evidence of improvement'
    return {'simulation': True, 'seed': SEED, 'sample_size_required_per_arm': required_n,
            'observed_n_a':na, 'observed_n_b':nb, 'alpha':ALPHA, 'target_power':POWER,
            'baseline_assumption':BASELINE, 'mde_absolute':MDE_ABSOLUTE,
            'daily_eligible_users_assumption':DAILY_USERS,
            'estimated_days':math.ceil(2*required_n/DAILY_USERS),
            'srm_chi2':srm_chi2, 'srm_p_value':srm_p, 'srm_pass':srm_ok,
            'sample_size_pass':sample_ok, 'primary':primary,
            'error_guardrail':{'rate_a':errors['rate_a'],'rate_b':errors['rate_b'],
                               'effect':errors['absolute_effect'],'p_value':errors['p_value'],
                               'ci95_upper':errors['ci95_upper'],
                               'maximum_acceptable_increase':GUARDRAIL_ERROR_MAX_INCREASE,'pass':error_ok},
            'duration_guardrail':{'mean_a':ta,'mean_b':tb,'effect_seconds':time_effect,
                                  'p_value':time_p,'ci95_upper_seconds':time_upper,
                                  'maximum_acceptable_increase_seconds':GUARDRAIL_TIME_MAX_INCREASE,
                                  'pass':time_ok}, 'decision':decision}


def make_charts(result):
    p = result['primary']
    fig, ax = plt.subplots(figsize=(7,4.5))
    bars = ax.bar(['A: 5 steps','B: 3 steps'],[p['rate_a']*100,p['rate_b']*100],color=['#087f8c','#ee964b'])
    ax.set_ylabel('Completion rate (%)')
    ax.set_title('Simulated A/B experiment: service completion')
    ax.set_ylim(0,100)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.2f}%',ha='center')
    fig.tight_layout()
    fig.savefig(CHARTS/'completion_rates.png',dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.errorbar([p['absolute_effect']*100],[0],
                xerr=[[100*(p['absolute_effect']-p['ci95_lower'])],
                      [100*(p['ci95_upper']-p['absolute_effect'])]],
                fmt='o',capsize=6,color='#087f8c')
    ax.axvline(0,color='gray',linestyle='--')
    ax.set_xlabel('Absolute effect B - A (percentage points)')
    ax.set_yticks([])
    ax.set_title('Estimated effect and 95% confidence interval')
    fig.tight_layout()
    fig.savefig(CHARTS/'effect_ci.png',dpi=160)
    plt.close(fig)


def main():
    required_n = sample_size_per_arm(BASELINE,BASELINE+MDE_ABSOLUTE,ALPHA,POWER)
    # Always regenerate to make the entire project reproducible.
    df = generate_data(required_n)
    result = analyze(df,required_n)
    make_charts(result)
    (RESULTS/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x: x.item() if hasattr(x, 'item') else str(x)),encoding='utf-8')
    lines = [f"Simulated A/B test | n per arm: {required_n}",
             f"Estimated duration: {result['estimated_days']} days at {DAILY_USERS} eligible users/day",
             f"SRM p-value: {result['srm_p_value']:.5g} (pass={result['srm_pass']})",
             f"Completion A: {result['primary']['rate_a']:.2%}",
             f"Completion B: {result['primary']['rate_b']:.2%}",
             f"Absolute effect: {result['primary']['absolute_effect']:.2%}",
             f"95% CI: [{result['primary']['ci95_lower']:.2%}, {result['primary']['ci95_upper']:.2%}]",
             f"p-value: {result['primary']['p_value']:.5g}",
             f"Error guardrail pass: {result['error_guardrail']['pass']}",
             f"Duration guardrail pass: {result['duration_guardrail']['pass']}",
             f"Decision: {result['decision']}",
             'NOTE: All observations are synthetic; no real-world causal claim is made.']
    report = '\n'.join(lines)+'\n'
    (RESULTS/'report.txt').write_text(report,encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
