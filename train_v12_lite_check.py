# -*- coding: utf-8 -*-
"""
Lightweight v12 checker for manual feature-diet loops.

Goals
1. Run a fast-ish validation close to v12.
2. Save CV and feature importance in one pass.
3. Let you re-run quickly with a manual drop list.

Examples
  python train_v12_lite_check.py
  python train_v12_lite_check.py --drop-file data/submissions/v12_lite_check/drop_cols.csv
  python train_v12_lite_check.py --models lgb cat --folds 3 --sample-frac 0.7
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import train_v12_final as v12


DEFAULT_OUTPUT_DIR = "./data/submissions/v12_lite_check"


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight v12 CV + importance checker")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgb"],
        choices=["lgb", "cat"],
        help="Models to run. Default is lgb only for speed.",
    )
    parser.add_argument("--folds", type=int, default=3, help="CV folds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Optional row sampling fraction")
    parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--drop-file", type=str, default="", help="CSV with a 'feature' column to drop before training")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--importance-topn", type=int, default=30, help="Top importance rows to print")
    parser.add_argument("--save-oof", action="store_true", help="Save OOF predictions")
    return parser.parse_args()


def maybe_sample(X, y, frac, seed):
    if frac >= 0.9999:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    sample_idx = (
        y.groupby(y, group_keys=False)
        .apply(lambda s: s.sample(frac=frac, random_state=seed))
        .index
    )
    sample_idx = np.array(sorted(sample_idx))
    return X.iloc[sample_idx].reset_index(drop=True), y.iloc[sample_idx].reset_index(drop=True)


def load_drop_features(path, available_features):
    if not path:
        return []
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in drop file: {path}")
    drops = [f for f in df["feature"].dropna().astype(str).tolist() if f in available_features]
    return sorted(set(drops))


class LGBWrapper:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, Xv, yv):
        params = {
            **v12.V11_LGB,
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "n_jobs": -1,
            "random_state": self.seed,
        }
        self.model = lgb.LGBMClassifier(**params, n_estimators=2500)
        self.model.fit(
            X,
            y,
            eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def importance(self):
        return self.model.feature_importances_


class CATWrapper:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, Xv, yv):
        params = {
            **v12.V8_CAT,
            "task_type": "CPU",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": self.seed,
        }
        self.model = CatBoostClassifier(**params, iterations=2500, early_stopping_rounds=100)
        self.model.fit(X, y, eval_set=(Xv, yv))

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def evaluate_lgb_with_importance(X, y, feat_cols, folds, seeds):
    all_oof = []
    total_importance = np.zeros(len(feat_cols))

    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))

        for fold, (ti, vi) in enumerate(skf.split(X, y), start=1):
            model = LGBWrapper(seed)
            model.fit(X.iloc[ti][feat_cols], y.iloc[ti], X.iloc[vi][feat_cols], y.iloc[vi])
            oof[vi] = model.predict(X.iloc[vi][feat_cols])
            total_importance += model.importance()
            print(f"  lgb fold {fold}/{folds} seed {seed} done")

        seed_cv = roc_auc_score(y, oof)
        print(f"  lgb seed {seed} cv: {seed_cv:.6f}")
        all_oof.append(oof)

    final_oof = np.mean(all_oof, axis=0)
    final_cv = roc_auc_score(y, final_oof)
    imp_df = pd.DataFrame({"feature": feat_cols, "importance": total_importance})
    imp_df = imp_df.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    return final_cv, final_oof, imp_df


def evaluate_cat(X, y, feat_cols, folds, seeds):
    all_oof = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))

        for fold, (ti, vi) in enumerate(skf.split(X, y), start=1):
            model = CATWrapper(seed)
            model.fit(X.iloc[ti][feat_cols], y.iloc[ti], X.iloc[vi][feat_cols], y.iloc[vi])
            oof[vi] = model.predict(X.iloc[vi][feat_cols])
            print(f"  cat fold {fold}/{folds} seed {seed} done")

        seed_cv = roc_auc_score(y, oof)
        print(f"  cat seed {seed} cv: {seed_cv:.6f}")
        all_oof.append(oof)

    final_oof = np.mean(all_oof, axis=0)
    final_cv = roc_auc_score(y, final_oof)
    return final_cv, final_oof


def save_drop_suggestions(imp_df, output_dir, ts):
    zero_imp = imp_df[imp_df["importance"] <= 0].copy()
    low_imp = imp_df.sort_values(["importance", "feature"], ascending=[True, True]).head(20).copy()

    zero_path = os.path.join(output_dir, f"v12_lite_zero_importance_{ts}.csv")
    low_path = os.path.join(output_dir, f"v12_lite_low_importance_top20_{ts}.csv")

    zero_imp[["feature"]].to_csv(zero_path, index=False, encoding="utf-8-sig")
    low_imp[["feature", "importance"]].to_csv(low_path, index=False, encoding="utf-8-sig")
    return zero_path, low_path, len(zero_imp)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("v12 lite check")
    print("=" * 60)
    print(f"models={args.models} folds={args.folds} seeds={args.seeds} sample_frac={args.sample_frac}")

    train, test, _ = v12.load_data()
    X_train, y_train, _, feature_cols = v12.preprocess(train, test)
    X_train, y_train = maybe_sample(X_train, y_train, args.sample_frac, args.sample_seed)

    drop_features = load_drop_features(args.drop_file, set(feature_cols))
    use_features = [f for f in feature_cols if f not in set(drop_features)]

    print(f"\nbase feature count : {len(feature_cols)}")
    print(f"drop feature count : {len(drop_features)}")
    print(f"final feature count: {len(use_features)}")

    if drop_features:
        print("\nDropped features:")
        for f in drop_features[:30]:
            print(f"  {f}")
        if len(drop_features) > 30:
            print(f"  ... and {len(drop_features) - 30} more")

    lgb_cv, lgb_oof, imp_df = evaluate_lgb_with_importance(X_train, y_train, use_features, args.folds, args.seeds)
    result_rows = [{"model": "lgb", "cv_auc": round(lgb_cv, 6)}]
    blend_oof = lgb_oof.copy()
    blend_cv = lgb_cv

    if "cat" in args.models:
        cat_cv, cat_oof = evaluate_cat(X_train, y_train, use_features, args.folds, args.seeds)
        result_rows.append({"model": "cat", "cv_auc": round(cat_cv, 6)})
        weights = np.array([lgb_cv, cat_cv], dtype=float)
        weights = weights / weights.sum()
        blend_oof = weights[0] * lgb_oof + weights[1] * cat_oof
        blend_cv = roc_auc_score(y_train, blend_oof)

    result_df = pd.DataFrame(result_rows)
    imp_path = os.path.join(args.output_dir, f"v12_lite_importance_{ts}.csv")
    summary_path = os.path.join(args.output_dir, f"v12_lite_summary_{ts}.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    zero_path, low_path, zero_cnt = save_drop_suggestions(imp_df, args.output_dir, ts)

    summary = pd.DataFrame([{
        "timestamp": ts,
        "models": " ".join(args.models),
        "folds": args.folds,
        "seeds": " ".join(map(str, args.seeds)),
        "sample_frac": args.sample_frac,
        "base_feature_count": len(feature_cols),
        "drop_feature_count": len(drop_features),
        "final_feature_count": len(use_features),
        "cv_lgb": round(lgb_cv, 6),
        "cv_blend": round(blend_cv, 6),
        "zero_importance_count": zero_cnt,
        "importance_file": imp_path,
        "zero_importance_file": zero_path,
        "low_importance_file": low_path,
        "drop_file": args.drop_file,
    }])
    if "cat" in args.models:
        summary["cv_cat"] = result_df.loc[result_df["model"] == "cat", "cv_auc"].iloc[0]
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nModel CV")
    print(result_df.to_string(index=False))
    print(f"\nblend cv: {blend_cv:.6f}")

    print(f"\nTop {args.importance_topn} importance")
    print(imp_df.head(args.importance_topn).to_string(index=False))

    print(f"\nBottom {args.importance_topn} importance")
    print(imp_df.tail(args.importance_topn).sort_values(["importance", "feature"], ascending=[True, True]).to_string(index=False))

    print("\nSaved files")
    print(f"  summary         : {summary_path}")
    print(f"  importance      : {imp_path}")
    print(f"  zero importance : {zero_path}")
    print(f"  low importance  : {low_path}")

    if args.save_oof:
        oof_path = os.path.join(args.output_dir, f"v12_lite_oof_{ts}.csv")
        oof_df = pd.DataFrame({"oof_lgb": lgb_oof, "y": y_train})
        if "cat" in args.models:
            oof_df["oof_blend"] = blend_oof
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
        print(f"  oof             : {oof_path}")


if __name__ == "__main__":
    main()
