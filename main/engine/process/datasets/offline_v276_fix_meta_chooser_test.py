import math
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).with_name('gaussian_fan_dataset_v2_2.csv')
LOOKBACK = 30
K_NEIGHBORS = 7
CHUNK_COUNT = 4

TRUTH_CANDIDATES = ['truth', 'true_direction', 'actual', 'label', 'target']
PRED_CANDIDATES = ['prediction', 'pred', 'trend', 'model_prediction', 'predicted_trend']
EPOCH_CANDIDATES = ['epoch', 'round', 'round_id', 'epoch_id', 'id']

RULEB_SLOW_THRESHOLD = 0.95
RULEB_HINGE_THRESHOLD = 0.15
RULEB_TRANSITION_MIN = 1


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'Could not find {label} column. Tried {candidates}. Available: {list(df.columns)}')


def sign_series(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(f'sign_series expected 1-D input, got {list(series.columns)}')
        series = series.iloc[:, 0]
    s = pd.to_numeric(series, errors='coerce').fillna(0.0)
    return ((s > 0).astype(int) - (s < 0).astype(int)).astype(int)


def compute_stats(df: pd.DataFrame, pred_col: str, truth_col: str) -> dict:
    total_rows = len(df)
    directional_mask = df[pred_col].isin(['Bull', 'Bear'])
    directional = df[directional_mask]
    directional_called = int(len(directional))
    neutral_called = int((df[pred_col] == 'Neutral').sum())
    directional_accuracy = float((directional[pred_col] == directional[truth_col]).mean()) if directional_called else 0.0
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0
    bull_pred = directional[directional[pred_col] == 'Bull']
    bear_pred = directional[directional[pred_col] == 'Bear']
    bull_precision = float((bull_pred[truth_col] == 'Bull').mean()) if len(bull_pred) else 0.0
    bear_precision = float((bear_pred[truth_col] == 'Bear').mean()) if len(bear_pred) else 0.0
    truth_bull = df[truth_col] == 'Bull'
    truth_bear = df[truth_col] == 'Bear'
    bull_recall = float((((df[pred_col] == 'Bull') & truth_bull).sum()) / truth_bull.sum()) if truth_bull.sum() else 0.0
    bear_recall = float((((df[pred_col] == 'Bear') & truth_bear).sum()) / truth_bear.sum()) if truth_bear.sum() else 0.0
    return {
        'rows': total_rows,
        'directional_called': directional_called,
        'neutral_called': neutral_called,
        'directional_accuracy': directional_accuracy,
        'directional_coverage': coverage,
        'neutral_rate': neutral_rate,
        'bull_precision': bull_precision,
        'bear_precision': bear_precision,
        'bull_recall': bull_recall,
        'bear_recall': bear_recall,
    }


def print_stats_block(title: str, stats: dict) -> None:
    print(f'\n=== {title} ===')
    print(f"Rows: {stats['rows']}")
    print(f"Directional Called: {stats['directional_called']}")
    print(f"Neutral Called: {stats['neutral_called']}")
    print(f"Directional Accuracy: {stats['directional_accuracy'] * 100:.2f}%")
    print(f"Directional Coverage: {stats['directional_coverage'] * 100:.2f}%")
    print(f"Neutral Rate: {stats['neutral_rate'] * 100:.2f}%")
    print(f"Bull Precision: {stats['bull_precision'] * 100:.2f}%")
    print(f"Bear Precision: {stats['bear_precision'] * 100:.2f}%")
    print(f"Bull Recall: {stats['bull_recall'] * 100:.2f}%")
    print(f"Bear Recall: {stats['bear_recall'] * 100:.2f}%")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [
        'g8', 'g23', 'g38', 'g53', 'g68', 'g83',
        'slope_g8', 'slope_g23', 'slope_g38', 'slope_g53', 'slope_g68', 'slope_g83',
        'hinge_torsion', 'fan_width_velocity', 'outer_fan_width',
        'phase_disagreement', 'fan_polarity_inversion', 'slow_retention',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    new_cols: dict[str, pd.Series] = {}
    if 'fan_energy_total' not in df.columns:
        new_cols['fan_energy_total'] = df['outer_fan_width'].abs() * (1.0 + df['hinge_torsion'].abs())
        fan_energy_total = new_cols['fan_energy_total']
    else:
        fan_energy_total = pd.to_numeric(df['fan_energy_total'], errors='coerce')

    new_cols['hinge_torsion_spike'] = (df['hinge_torsion'].abs() > 0.5).astype(int)
    new_cols['transition_score_v23'] = (
        (df['phase_disagreement'] == 1).astype(int)
        + new_cols['hinge_torsion_spike']
        + (df['fan_polarity_inversion'] == 1).astype(int)
    )
    new_cols['fan_energy_t0'] = fan_energy_total
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def rolling_support(series: pd.Series, mode: str) -> pd.Series:
    s = series.fillna(0).astype(int)
    if mode == 'pm1':
        return ((s == 1) | (s.shift(1).fillna(0) == 1) | (s.shift(-1).fillna(0) == 1)).astype(int)
    raise ValueError(f'Unsupported rolling support mode: {mode}')


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols: dict[str, pd.Series] = {}

    fast_now = sign_series(df['slope_g8'] + df['slope_g23'])
    slow_now = sign_series(df['slope_g68'] + df['slope_g83'])
    cols['fast_sign_now'] = fast_now
    cols['slow_sign_now'] = slow_now
    cols['fs_disagree_now'] = (fast_now != slow_now).astype(int)
    cols['fs_disagree_count_3'] = (
        cols['fs_disagree_now']
        + cols['fs_disagree_now'].shift(1).fillna(0)
        + cols['fs_disagree_now'].shift(2).fillna(0)
    ).astype(int)

    fast_t1 = fast_now.shift(1).fillna(0).astype(int)
    fast_t2 = fast_now.shift(2).fillna(0).astype(int)
    cols['fast_alt_3epoch'] = (
        (fast_now != 0)
        & (fast_t1 != 0)
        & (fast_t2 != 0)
        & (fast_now == fast_t2)
        & (fast_now != fast_t1)
    ).astype(int)

    phase = pd.to_numeric(df['phase_disagreement'], errors='coerce').fillna(0).astype(int)
    polarity = pd.to_numeric(df['fan_polarity_inversion'], errors='coerce').fillna(0).astype(int)
    cols['phase_disagree_count_3'] = (phase + phase.shift(1).fillna(0) + phase.shift(2).fillna(0)).astype(int)
    cols['polarity_inv_count_3'] = (polarity + polarity.shift(1).fillna(0) + polarity.shift(2).fillna(0)).astype(int)

    fan_energy_total = df['fan_energy_total']
    if isinstance(fan_energy_total, pd.DataFrame):
        fan_energy_total = fan_energy_total.iloc[:, 0]
    fan_energy_total = pd.to_numeric(fan_energy_total, errors='coerce')
    energy_delta = fan_energy_total.diff().fillna(0.0)
    cols['energy_delta'] = energy_delta
    cols['energy_sign'] = sign_series(energy_delta)
    cols['energy_flip_count_3'] = (
        (cols['energy_sign'] != cols['energy_sign'].shift(1)).fillna(False).astype(int)
        + (cols['energy_sign'].shift(1) != cols['energy_sign'].shift(2)).fillna(False).astype(int)
    ).astype(int)

    width_sign = sign_series(df['fan_width_velocity'])
    cols['width_rebound_pattern_3'] = (
        (width_sign != 0)
        & (width_sign.shift(1).fillna(0) != 0)
        & (width_sign.shift(2).fillna(0) != 0)
        & (width_sign == width_sign.shift(2).fillna(0))
        & (width_sign != width_sign.shift(1).fillna(0))
    ).astype(int)

    cols['osc_core_phase'] = (cols['phase_disagree_count_3'] >= 2).astype(int)
    cols['osc_core_fs'] = (cols['fs_disagree_count_3'] >= 2).astype(int)
    cols['osc_core_fastalt'] = cols['fast_alt_3epoch'].astype(int)
    cols['osc_core_count'] = (cols['osc_core_phase'] + cols['osc_core_fs'] + cols['osc_core_fastalt']).astype(int)
    cols['osc_core_a'] = (cols['osc_core_count'] >= 2).astype(int)

    cols['center_any_support'] = (
        (cols['phase_disagree_count_3'] >= 1)
        | (cols['fs_disagree_count_3'] >= 1)
        | (cols['fast_alt_3epoch'] == 1)
    ).astype(int)
    cols['center_phase_or_fs'] = (
        (cols['phase_disagree_count_3'] >= 1)
        | (cols['fs_disagree_count_3'] >= 1)
    ).astype(int)

    cols['phase_support_pm1'] = rolling_support(cols['osc_core_phase'], 'pm1')
    cols['fs_support_pm1'] = rolling_support(cols['osc_core_fs'], 'pm1')
    cols['fastalt_support_pm1'] = rolling_support(cols['osc_core_fastalt'], 'pm1')
    cols['core_window_pm1'] = (
        (cols['fs_support_pm1'] == 1)
        & ((cols['phase_support_pm1'] == 1) | (cols['fastalt_support_pm1'] == 1))
    ).astype(int)
    cols['pm1_centered_soft'] = (
        (cols['core_window_pm1'] == 1)
        & ((cols['center_phase_or_fs'] == 1) | (cols['osc_core_fastalt'] == 1))
        & (df['hinge_torsion'].abs() >= 0.20)
    ).astype(int)
    cols['pm1_hinge_018'] = (
        (cols['core_window_pm1'] == 1)
        & (cols['center_any_support'] == 1)
        & (df['hinge_torsion'].abs() >= 0.18)
    ).astype(int)

    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def build_ruleb_warning(df: pd.DataFrame) -> pd.Series:
    return (
        (df['slow_retention'] > RULEB_SLOW_THRESHOLD)
        & (df['hinge_torsion'].abs() > RULEB_HINGE_THRESHOLD)
        & (df['transition_score_v23'] >= RULEB_TRANSITION_MIN)
    ).astype(int)


def apply_bear_variant(df: pd.DataFrame, base_col: str, trigger_mask: pd.Series, variant_name: str, truth_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    base_pred = df[base_col].copy()
    # Locked Bear-side methods must require BOTH a Bear base prediction AND RuleB raw context.
    eligible = (base_pred == 'Bear') & (df['ruleb_raw_warning'] == 1)
    trigger = eligible & trigger_mask.fillna(False)
    pred = base_pred.where(~trigger, 'Neutral')
    out[f'pred_{variant_name}'] = pred
    out[f'trigger_{variant_name}'] = trigger.astype(int)
    base_correct = df[base_col] == df[truth_col]
    override_class = pd.Series('NO_EFFECT', index=df.index, dtype='object')
    override_class.loc[trigger & (~base_correct)] = 'GOOD_NEUTRALIZE_WRONG_BASE'
    override_class.loc[trigger & base_correct] = 'BAD_NEUTRALIZE_RIGHT_BASE'
    out[f'override_help_class_{variant_name}'] = override_class
    return out


def trigger_quality(df: pd.DataFrame, truth_col: str, base_col: str, pred_col: str, trigger_col: str, label: str) -> dict:
    trigger = df[trigger_col] == 1
    base_correct = df[base_col] == df[truth_col]
    good = int((trigger & (~base_correct)).sum())
    bad = int((trigger & base_correct).sum())
    return {
        'model': label,
        'triggers': int(trigger.sum()),
        'good': good,
        'bad': bad,
        'good_rate': (good / int(trigger.sum())) if int(trigger.sum()) else 0.0,
    }


def action_utility(row: pd.Series, action: str, truth_col: str) -> float:
    pred = row[f'pred_{action}'] if action != 'base' else row['pred_v22_base']
    truth = row[truth_col]
    if pred == 'Neutral':
        return 1.0 if truth == 'Bull' else -1.0
    return 1.0 if pred == truth else -1.0


def weighted_action_scores(hist: pd.DataFrame, cur: pd.Series, truth_col: str) -> dict[str, float]:
    features = ['slow_retention', 'abs_hinge', 'transition_score_v23', 'phase_disagree_count_3', 'fs_disagree_count_3', 'fast_alt_3epoch', 'center_any_support', 'core_window_pm1']
    H = hist[features].copy()
    means = H.mean()
    stds = H.std(ddof=0).replace(0, 1.0)
    zhist = (H - means) / stds
    zcur = ((cur[features] - means) / stds).astype(float)
    dists = np.sqrt(((zhist - zcur) ** 2).sum(axis=1).values)
    k = min(K_NEIGHBORS, len(hist))
    order = np.argsort(dists)[:k]
    neigh = hist.iloc[order].copy()
    weights = 1.0 / (1.0 + dists[order])

    scores = {}
    for action in ['base', 'corea', 'fallback', 'winner']:
        utils = np.array([action_utility(r, action, truth_col) for _, r in neigh.iterrows()], dtype=float)
        score = float(np.average(utils, weights=weights)) if len(utils) else 0.0
        # Tiny complexity penalty to prefer simpler actions in near ties
        if action == 'winner':
            score -= 0.01
        elif action == 'fallback':
            score -= 0.005
        scores[action] = score
    return scores


def build_meta_chooser(df: pd.DataFrame, truth_col: str) -> pd.DataFrame:
    policies = []
    reasons = []
    preds = []
    triggers = []

    for i in range(len(df)):
        row = df.iloc[i]
        base_pred = row['pred_v22_base']
        if i < LOOKBACK:
            policy = 'base'
            reason = f'warmup_lt_{LOOKBACK}'
        elif base_pred != 'Bear':
            policy = 'base'
            reason = f'base_pred={base_pred}'
        elif int(row['ruleb_raw_warning']) != 1:
            policy = 'base'
            reason = 'ruleb_off'
        else:
            hist = df.iloc[max(0, i - LOOKBACK):i].copy()
            eligible = hist[(hist['pred_v22_base'] == 'Bear') & (hist['ruleb_raw_warning'] == 1)].copy()
            if len(eligible) < 5:
                policy = 'base'
                reason = 'meta_not_enough_neighbors'
            else:
                scores = weighted_action_scores(eligible, row, truth_col)
                policy = max(scores, key=scores.get)
                margin = scores[policy] - scores['base']
                if margin < 0.05:
                    policy = 'base'
                    reason = f"meta_no_edge|base={scores['base']:.3f}|corea={scores['corea']:.3f}|fallback={scores['fallback']:.3f}|winner={scores['winner']:.3f}"
                else:
                    reason = f"meta_pick={policy}|base={scores['base']:.3f}|corea={scores['corea']:.3f}|fallback={scores['fallback']:.3f}|winner={scores['winner']:.3f}"

        if policy == 'winner':
            pred = row['pred_winner']
            trig = int(row['trigger_winner'])
        elif policy == 'fallback':
            pred = row['pred_fallback']
            trig = int(row['trigger_fallback'])
        elif policy == 'corea':
            pred = row['pred_corea']
            trig = int(row['trigger_corea'])
        else:
            pred = row['pred_v22_base']
            trig = 0

        policies.append(policy)
        reasons.append(reason)
        preds.append(pred)
        triggers.append(trig)

    out = pd.DataFrame(index=df.index)
    out['adaptive_policy_meta'] = pd.Series(policies, index=df.index, dtype='object')
    out['adaptive_reason_meta'] = pd.Series(reasons, index=df.index, dtype='object')
    out['pred_adaptive_meta'] = pd.Series(preds, index=df.index, dtype='object')
    out['trigger_adaptive_meta'] = pd.Series(triggers, index=df.index, dtype='int64')
    return out


def capture_of_wrong_bear(df: pd.DataFrame, truth_col: str, trigger_col: str) -> float:
    universe = (df['pred_v22_base'] == 'Bear') & (df['ruleb_raw_warning'] == 1) & (df[truth_col] == 'Bull')
    denom = int(universe.sum())
    if denom == 0:
        return 0.0
    return float(((df[trigger_col] == 1) & universe).sum() / denom)


def chunk_stability(df: pd.DataFrame, truth_col: str, specs: list[tuple[str, str]]) -> pd.DataFrame:
    chunk_size = math.ceil(len(df) / CHUNK_COUNT)
    rows = []
    for chunk_id in range(CHUNK_COUNT):
        start = chunk_id * chunk_size
        end = min(len(df), (chunk_id + 1) * chunk_size)
        sub = df.iloc[start:end]
        for label, pred_col in specs:
            stats = compute_stats(sub, pred_col, truth_col)
            trig_col = pred_col.replace('pred_', 'trigger_') if pred_col.startswith('pred_') else None
            trig = int(sub[trig_col].sum()) if trig_col and trig_col in sub.columns else 0
            rows.append({
                'chunk': chunk_id + 1,
                'model': label,
                'rows': len(sub),
                'trig': trig,
                'acc': stats['directional_accuracy'],
                'cov': stats['directional_coverage'],
                'bear': stats['bear_precision'],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    truth_col = pick_column(df, TRUTH_CANDIDATES, 'truth')
    base_input_col = pick_column(df, PRED_CANDIDATES, 'base prediction')
    epoch_col = pick_column(df, EPOCH_CANDIDATES, 'epoch id')

    df = df.rename(columns={truth_col: 'truth', base_input_col: 'pred_v22_base', epoch_col: 'epoch'})
    df = build_features(df)
    df['abs_hinge'] = df['hinge_torsion'].abs()
    df = build_sequence_features(df)
    df['ruleb_raw_warning'] = build_ruleb_warning(df)

    corea = apply_bear_variant(df, 'pred_v22_base', df['osc_core_a'] == 1, 'corea', 'truth')
    fallback = apply_bear_variant(df, 'pred_v22_base', df['pm1_centered_soft'] == 1, 'fallback', 'truth')
    winner = apply_bear_variant(df, 'pred_v22_base', df['pm1_hinge_018'] == 1, 'winner', 'truth')
    df = pd.concat([df, corea, fallback, winner], axis=1)

    adaptive = build_meta_chooser(df, 'truth')
    df = pd.concat([df, adaptive], axis=1)

    last = df.iloc[-1]
    print('\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===')
    print(f"Base v2_2 prediction: {last['pred_v22_base']}")
    print(f"slow_retention: {float(last['slow_retention']):.4f}")
    print(f"|hinge_torsion|: {abs(float(last['hinge_torsion'])):.4f}")
    print(f"transition_score_v23: {int(last['transition_score_v23'])}")
    print(f"ruleb_raw_warning: {int(last['ruleb_raw_warning'])}")
    print(f"phase_disagree_count_3: {int(last['phase_disagree_count_3'])}")
    print(f"fs_disagree_count_3: {int(last['fs_disagree_count_3'])}")
    print(f"fast_alt_3epoch: {int(last['fast_alt_3epoch'])}")
    print(f"pm1_centered_soft: {int(last['pm1_centered_soft'])}")
    print(f"pm1_hinge_018: {int(last['pm1_hinge_018'])}")
    print(f"Adaptive chosen policy: {last['adaptive_policy_meta']}")
    print(f"Adaptive reason: {last['adaptive_reason_meta']}")

    specs = [
        ('v2_2 (base)', 'pred_v22_base'),
        ('Safe reference: Bear-only RuleB + Core A', 'pred_corea'),
        ('Fallback: Bear-only RuleB + Center Soft', 'pred_fallback'),
        ('Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18', 'pred_winner'),
        ('Adaptive v276_fix: KNN meta-chooser (30-epoch lookback)', 'pred_adaptive_meta'),
    ]
    for label, pred_col in specs:
        print_stats_block(label, compute_stats(df, pred_col, 'truth'))

    summary_rows = []
    trigger_rows = []
    for label, pred_col, trig_col in [
        ('bear_corea', 'pred_corea', 'trigger_corea'),
        ('bear_pm1_soft', 'pred_fallback', 'trigger_fallback'),
        ('bear_pm1_h018', 'pred_winner', 'trigger_winner'),
        ('adaptive_meta', 'pred_adaptive_meta', 'trigger_adaptive_meta'),
    ]:
        stats = compute_stats(df, pred_col, 'truth')
        tq = trigger_quality(df, 'truth', 'pred_v22_base', pred_col, trig_col, label)
        capture = capture_of_wrong_bear(df, 'truth', trig_col)
        summary_rows.append({
            'model': label,
            'display_name': dict(specs)[next(k for k,v in specs if v == pred_col)],
            'acc': stats['directional_accuracy'],
            'cov': stats['directional_coverage'],
            'bear': stats['bear_precision'],
            'trig': tq['triggers'],
            'good': tq['good'],
            'bad': tq['bad'],
            'capture': capture,
        })
        trigger_rows.append({**tq})

    # Add base to summary
    base_stats = compute_stats(df, 'pred_v22_base', 'truth')
    summary_rows.append({
        'model': 'pred_v22_base', 'display_name': 'v2_2 (base)', 'acc': base_stats['directional_accuracy'],
        'cov': base_stats['directional_coverage'], 'bear': base_stats['bear_precision'], 'trig': 0, 'good': 0, 'bad': 0, 'capture': 0.0,
    })
    summary_df = pd.DataFrame(summary_rows).sort_values(['acc', 'bear', 'cov'], ascending=False).reset_index(drop=True)
    trigger_df = pd.DataFrame(trigger_rows)

    print('\n=== V276 META-CHOOSER LEADERBOARD ===')
    for _, row in summary_df.iterrows():
        print(f"{row['model']} | {row['display_name']} | acc={row['acc']*100:.2f}% | cov={row['cov']*100:.2f}% | bear={row['bear']*100:.2f}% | trig={int(row['trig'])} | good={int(row['good'])} | bad={int(row['bad'])} | capture={row['capture']*100:.2f}%")

    print('\n=== V276 TRIGGER QUALITY SUMMARY ===')
    for _, row in trigger_df.iterrows():
        print(f"{row['model']} | triggers={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])} | good_rate={row['good_rate']*100:.2f}%")

    policy_usage = df.groupby('adaptive_policy_meta', dropna=False).apply(lambda g: pd.Series({
        'rows': len(g),
        'bear_rows': int((g['pred_v22_base'] == 'Bear').sum()),
        'triggers': int(g['trigger_adaptive_meta'].sum()),
        'good': int(((g['trigger_adaptive_meta'] == 1) & (g['pred_v22_base'] != g['truth'])).sum()),
        'bad': int(((g['trigger_adaptive_meta'] == 1) & (g['pred_v22_base'] == g['truth'])).sum()),
    })).reset_index().rename(columns={'adaptive_policy_meta': 'policy'})
    print('\n=== ADAPTIVE POLICY USAGE SUMMARY ===')
    for _, row in policy_usage.iterrows():
        print(f"policy={row['policy']} | rows={int(row['rows'])} | bear_rows={int(row['bear_rows'])} | triggers={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])}")

    reason_usage = df.groupby('adaptive_reason_meta', dropna=False).apply(lambda g: pd.Series({
        'rows': len(g), 'triggers': int(g['trigger_adaptive_meta'].sum()),
        'good': int(((g['trigger_adaptive_meta'] == 1) & (g['pred_v22_base'] != g['truth'])).sum()),
        'bad': int(((g['trigger_adaptive_meta'] == 1) & (g['pred_v22_base'] == g['truth'])).sum()),
    })).reset_index().rename(columns={'adaptive_reason_meta': 'reason'}).sort_values(['triggers', 'rows'], ascending=False)
    print('\n=== ADAPTIVE REASON SUMMARY ===')
    for _, row in reason_usage.head(12).iterrows():
        print(f"reason={row['reason']} | rows={int(row['rows'])} | triggers={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])}")

    chunk_df = chunk_stability(df, 'truth', [
        ('base', 'pred_v22_base'),
        ('winner', 'pred_winner'),
        ('fallback', 'pred_fallback'),
        ('adaptive_meta', 'pred_adaptive_meta'),
    ])
    print('\n=== ADAPTIVE VS STATIC CHUNK STABILITY ===')
    for _, row in chunk_df.iterrows():
        print(f"chunk={int(row['chunk'])} | model={row['model']} | rows={int(row['rows'])} | trig={int(row['trig'])} | acc={row['acc']*100:.2f}% | cov={row['cov']*100:.2f}% | bear={row['bear']*100:.2f}%")

    overlap_rows = []
    masks = {
        'winner_only': (df['trigger_winner'] == 1) & (df['trigger_fallback'] == 0) & (df['trigger_adaptive_meta'] == 0),
        'fallback_only': (df['trigger_fallback'] == 1) & (df['trigger_winner'] == 0) & (df['trigger_adaptive_meta'] == 0),
        'adaptive_only': (df['trigger_adaptive_meta'] == 1) & (df['trigger_winner'] == 0) & (df['trigger_fallback'] == 0),
        'adaptive_and_winner': (df['trigger_adaptive_meta'] == 1) & (df['trigger_winner'] == 1),
        'adaptive_and_fallback': (df['trigger_adaptive_meta'] == 1) & (df['trigger_fallback'] == 1),
        'adaptive_any': (df['trigger_adaptive_meta'] == 1),
    }
    for label, mask in masks.items():
        sub = df[mask]
        overlap_rows.append({
            'bucket': label,
            'rows': len(sub),
            'wrong_bear': int(((sub['pred_v22_base'] == 'Bear') & (sub['truth'] == 'Bull')).sum()),
            'right_bear': int(((sub['pred_v22_base'] == 'Bear') & (sub['truth'] == 'Bear')).sum()),
            'mean_slow': float(sub['slow_retention'].mean()) if len(sub) else 0.0,
            'mean_abs_hinge': float(sub['abs_hinge'].mean()) if len(sub) else 0.0,
        })
    overlap_df = pd.DataFrame(overlap_rows)
    print('\n=== WINNER/FALLBACK/ADAPTIVE OVERLAP SUMMARY ===')
    for _, row in overlap_df.iterrows():
        print(f"{row['bucket']} | rows={int(row['rows'])} | wrong_bear={int(row['wrong_bear'])} | right_bear={int(row['right_bear'])} | mean_slow={row['mean_slow']:.3f} | mean_|hinge|={row['mean_abs_hinge']:.3f}")

    remaining_fn = df[(df['pred_v22_base'] == 'Bear') & (df['truth'] == 'Bull') & (df['ruleb_raw_warning'] == 1) & (df['trigger_adaptive_meta'] == 0)].copy()
    print('\n=== REMAINING FALSE NEGATIVE AUDIT (MISSED BY ADAPTIVE) ===')
    for _, r in remaining_fn.iterrows():
        print(
            f"epoch={r['epoch']} | base=Bear | truth={r['truth']} | slow={float(r['slow_retention']):.3f} | |hinge|={float(r['abs_hinge']):.3f} | "
            f"trans={int(r['transition_score_v23'])} | phase3={int(r['phase_disagree_count_3'])} | fs3={int(r['fs_disagree_count_3'])} | "
            f"fast_alt={int(r['fast_alt_3epoch'])} | core_count={int(r['osc_core_count'])} | policy={r['adaptive_policy_meta']} | reason={r['adaptive_reason_meta']}"
        )

    out_dir = CSV_PATH.parent
    df.to_csv(out_dir / 'gaussian_fan_dataset_v2_2_with_v276_fix_meta_diagnostics.csv', index=False)
    df.to_csv(out_dir / 'offline_v276_fix_decision_audit.csv', index=False)
    trigger_df.to_csv(out_dir / 'offline_v276_fix_trigger_quality_summary.csv', index=False)
    summary_df.to_csv(out_dir / 'offline_v276_fix_rollout_summary.csv', index=False)
    policy_usage.to_csv(out_dir / 'offline_v276_fix_policy_usage_summary.csv', index=False)
    reason_usage.to_csv(out_dir / 'offline_v276_fix_reason_summary.csv', index=False)
    chunk_df.to_csv(out_dir / 'offline_v276_fix_chunk_stability.csv', index=False)
    overlap_df.to_csv(out_dir / 'offline_v276_fix_overlap_report.csv', index=False)
    remaining_fn.to_csv(out_dir / 'offline_v276_fix_remaining_false_negative_audit.csv', index=False)
    summary_df.to_csv(out_dir / 'offline_v276_fix_model_comparison.csv', index=False)

    print(f"\nSaved dataset: {out_dir / 'gaussian_fan_dataset_v2_2_with_v276_fix_meta_diagnostics.csv'}")
    print(f"Saved decision audit: {out_dir / 'offline_v276_fix_decision_audit.csv'}")
    print(f"Saved trigger quality summary: {out_dir / 'offline_v276_fix_trigger_quality_summary.csv'}")
    print(f"Saved rollout summary: {out_dir / 'offline_v276_fix_rollout_summary.csv'}")
    print(f"Saved policy usage summary: {out_dir / 'offline_v276_fix_policy_usage_summary.csv'}")
    print(f"Saved reason summary: {out_dir / 'offline_v276_fix_reason_summary.csv'}")
    print(f"Saved chunk stability: {out_dir / 'offline_v276_fix_chunk_stability.csv'}")
    print(f"Saved overlap report: {out_dir / 'offline_v276_fix_overlap_report.csv'}")
    print(f"Saved remaining false negative audit: {out_dir / 'offline_v276_fix_remaining_false_negative_audit.csv'}")
    print(f"Saved model comparison: {out_dir / 'offline_v276_fix_model_comparison.csv'}")


if __name__ == '__main__':
    main()
