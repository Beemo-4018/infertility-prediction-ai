# -*- coding: utf-8 -*-
"""
Iterative feature-diet search for the v8plus pipeline.

Workflow
1. Build the full v8plus feature matrix.
2. Remove low-importance features one by one while the best ensemble CV improves.
3. Once the CV stops improving, try adding back a curated set of core derived features one by one.
4. Save the experiment log, best feature list, best feature importance, and final submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import train_v8plus as v8


DEFAULT_IMPORTANCE_PATH = Path("submissions/feature_importance_v8plus_cpu.csv")
DEFAULT_OUTPUT_DIR = Path("submissions/feature_diet_search")

CORE_DERIVED_CANDIDATES = [
    "클리닉_나이별성공률",
    "클리닉_시술유형별성공률",
    "클리닉대비_개인성공률차이",
    "시술시기코드_성공률",
    "시술시기코드_시술건수",
    "시술시기코드_성공률편차",
    "시술시기코드_배아이식수평균",
    "배아_생성률",
    "배아_활용률",
    "배아_이식비율",
    "배아_저장비율",
    "난자_수정률",
    "미세주입_성공률",
    "미세주입_이식률",
    "미세주입후_저장비율",
    "클리닉_집중도",
    "과거_임신성공률",
    "과거_출산성공률",
    "IVF_임신성공률",
    "IVF_출산성공률",
    "DI_임신성공률",
    "IVF_비율",
    "failure_streak",
    "파트너정자_비율",
    "해동난자_비율",
    "시술유형_나이조합_te",
    "시술유형_불임주원인조합_te",
    "채취_이식_간격",
    "혼합_이식_간격",
    "해동_이식_간격",
    "배반포_이식추정",
    "남성_불임원인_수",
    "여성_불임원인_수",
    "총_불임원인_수",
    "고령_동결배아조합",
    "초고령_반복시술",
    "기증난자_고령조합",
    "기증난자_젊음_고령모",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative v8plus feature-diet search."
    )
    parser.add_argument(
        "--importance-path",
        type=Path,
        default=DEFAULT_IMPORTANCE_PATH,
        help="Path to the baseline v8plus feature-importance CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for logs and outputs.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use the GPU branches in the imported v8plus training functions.",
    )
    parser.add_argument(
        "--run-optuna-once",
        action="store_true",
        help="Tune LightGBM once on the full feature set and reuse the params.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials when --run-optuna-once is enabled.",
    )
    parser.add_argument(
        "--min-improve",
        type=float,
        default=1e-5,
        help="Minimum CV gain required to accept a prune/add step.",
    )
    parser.add_argument(
        "--max-prune-steps",
        type=int,
        default=None,
        help="Optional cap on accepted prune steps.",
    )
    parser.add_argument(
        "--max-add-steps",
        type=int,
        default=None,
        help="Optional cap on forward-add attempts.",
    )
    parser.add_argument(
        "--allow-adding-unremoved-core",
        action="store_true",
        help="Also test core derived features that were never removed during pruning.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_feature_priority(
    importance_path: Path,
    available_features: Sequence[str],
) -> pd.DataFrame:
    importance = pd.read_csv(importance_path)
    if "feature" not in importance.columns or "importance" not in importance.columns:
        raise ValueError(f"Unexpected importance schema: {importance_path}")

    feature_set = set(available_features)
    importance = importance[importance["feature"].isin(feature_set)].copy()
    missing = sorted(feature_set.difference(importance["feature"]))
    if missing:
        importance = pd.concat(
            [
                importance,
                pd.DataFrame({"feature": missing, "importance": [float("inf")] * len(missing)}),
            ],
            ignore_index=True,
        )
    return importance.sort_values(["importance", "feature"], ascending=[True, True]).reset_index(drop=True)


def evaluate_ensemble(
    y_true,
    oof_lgb,
    oof_xgb,
    oof_cat,
    pred_lgb,
    pred_xgb,
    pred_cat,
    cv_lgb: float,
    cv_xgb: float,
    cv_cat: float,
) -> Tuple[str, float, pd.DataFrame, pd.Series]:
    weights = pd.Series([cv_lgb, cv_xgb, cv_cat], index=["lgb", "xgb", "cat"], dtype=float)
    weights = weights / weights.sum()

    r_lgb = rankdata(oof_lgb) / len(oof_lgb)
    r_xgb = rankdata(oof_xgb) / len(oof_xgb)
    r_cat = rankdata(oof_cat) / len(oof_cat)
    rt_lgb = rankdata(pred_lgb) / len(pred_lgb)
    rt_xgb = rankdata(pred_xgb) / len(pred_xgb)
    rt_cat = rankdata(pred_cat) / len(pred_cat)

    candidates: Dict[str, Tuple[float, pd.Series]] = {
        "단순평균": (
            roc_auc_score(y_true, (oof_lgb + oof_xgb + oof_cat) / 3),
            pd.Series((pred_lgb + pred_xgb + pred_cat) / 3),
        ),
        "성능가중": (
            roc_auc_score(
                y_true,
                weights["lgb"] * oof_lgb + weights["xgb"] * oof_xgb + weights["cat"] * oof_cat,
            ),
            pd.Series(
                weights["lgb"] * pred_lgb + weights["xgb"] * pred_xgb + weights["cat"] * pred_cat
            ),
        ),
        "LGB강조(0.5/0.25/0.25)": (
            roc_auc_score(y_true, 0.5 * oof_lgb + 0.25 * oof_xgb + 0.25 * oof_cat),
            pd.Series(0.5 * pred_lgb + 0.25 * pred_xgb + 0.25 * pred_cat),
        ),
        "LGB+CAT(0.6/0.4)": (
            roc_auc_score(y_true, 0.6 * oof_lgb + 0.4 * oof_cat),
            pd.Series(0.6 * pred_lgb + 0.4 * pred_cat),
        ),
        "LGB+CAT(0.5/0.5)": (
            roc_auc_score(y_true, 0.5 * oof_lgb + 0.5 * oof_cat),
            pd.Series(0.5 * pred_lgb + 0.5 * pred_cat),
        ),
        "LGB단독": (cv_lgb, pd.Series(pred_lgb)),
        "Rank단순평균": (
            roc_auc_score(y_true, (r_lgb + r_xgb + r_cat) / 3),
            pd.Series((rt_lgb + rt_xgb + rt_cat) / 3),
        ),
        "Rank성능가중": (
            roc_auc_score(
                y_true,
                weights["lgb"] * r_lgb + weights["xgb"] * r_xgb + weights["cat"] * r_cat,
            ),
            pd.Series(
                weights["lgb"] * rt_lgb + weights["xgb"] * rt_xgb + weights["cat"] * rt_cat
            ),
        ),
        "Rank LGB강조(0.5/0.25/0.25)": (
            roc_auc_score(y_true, 0.5 * r_lgb + 0.25 * r_xgb + 0.25 * r_cat),
            pd.Series(0.5 * rt_lgb + 0.25 * rt_xgb + 0.25 * rt_cat),
        ),
        "Rank LGB+CAT(0.6/0.4)": (
            roc_auc_score(y_true, 0.6 * r_lgb + 0.4 * r_cat),
            pd.Series(0.6 * rt_lgb + 0.4 * rt_cat),
        ),
        "Rank LGB+CAT(0.5/0.5)": (
            roc_auc_score(y_true, 0.5 * r_lgb + 0.5 * r_cat),
            pd.Series(0.5 * rt_lgb + 0.5 * rt_cat),
        ),
    }

    score_table = pd.DataFrame(
        [{"ensemble": name, "cv_auc": score_pred[0]} for name, score_pred in candidates.items()]
    ).sort_values("cv_auc", ascending=False, ignore_index=True)

    best_name = score_table.iloc[0]["ensemble"]
    best_score, best_pred = candidates[best_name]
    return best_name, float(best_score), score_table, best_pred


def evaluate_feature_set(
    step_name: str,
    feature_names: Sequence[str],
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    X_test_full: pd.DataFrame,
    best_params: Optional[dict],
) -> dict:
    print("\n" + "=" * 80)
    print(f"[{step_name}] evaluating {len(feature_names)} features")
    print("=" * 80)

    X_train = X_train_full.loc[:, feature_names]
    X_test = X_test_full.loc[:, feature_names]

    oof_lgb, pred_lgb, cv_lgb, importance = v8.train_lgb(
        X_train, y_train, X_test, list(feature_names), best_params
    )
    oof_xgb, pred_xgb, cv_xgb = v8.train_xgb(X_train, y_train, X_test)
    oof_cat, pred_cat, cv_cat = v8.train_cat(X_train, y_train, X_test)

    best_ensemble, best_cv, ensemble_scores, best_pred = evaluate_ensemble(
        y_train.values,
        oof_lgb,
        oof_xgb,
        oof_cat,
        pred_lgb,
        pred_xgb,
        pred_cat,
        cv_lgb,
        cv_xgb,
        cv_cat,
    )

    print(f"best ensemble: {best_ensemble} / CV={best_cv:.5f}")
    return {
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "cv_lgb": float(cv_lgb),
        "cv_xgb": float(cv_xgb),
        "cv_cat": float(cv_cat),
        "best_ensemble": best_ensemble,
        "best_cv": float(best_cv),
        "ensemble_scores": ensemble_scores,
        "best_pred": best_pred,
        "importance": importance,
    }


def save_feature_list(path: Path, feature_names: Sequence[str]) -> None:
    pd.DataFrame({"feature": list(feature_names)}).to_csv(path, index=False)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    v8.USE_GPU = args.use_gpu
    v8.USE_OPTUNA = False

    train, test, sub = v8.load_data()
    raw_columns = set(train.columns).union(test.columns)
    X_train_full, y_train, X_test_full, feature_cols = v8.preprocess(train, test)
    derived_features = [feature for feature in feature_cols if feature not in raw_columns]

    importance = load_feature_priority(args.importance_path, feature_cols)
    prune_priority = importance["feature"].tolist()

    best_params = None
    if args.run_optuna_once:
        print("\nRunning one-time Optuna tuning on the full feature set...")
        best_params = v8.optuna_lgb(X_train_full, y_train, n_trials=args.optuna_trials)

    experiment_log: List[dict] = []

    baseline = evaluate_feature_set(
        step_name="baseline",
        feature_names=feature_cols,
        X_train_full=X_train_full,
        y_train=y_train,
        X_test_full=X_test_full,
        best_params=best_params,
    )
    experiment_log.append(
        {
            "stage": "baseline",
            "status": "accepted",
            "feature": "",
            "feature_count": baseline["feature_count"],
            "best_cv": baseline["best_cv"],
            "best_ensemble": baseline["best_ensemble"],
            "delta_vs_best": 0.0,
        }
    )

    active_features = list(feature_cols)
    best_result = baseline
    removed_features: List[str] = []
    prune_stop_feature: Optional[str] = None
    accepted_prunes = 0

    for feature in prune_priority:
        if feature not in active_features:
            continue
        if args.max_prune_steps is not None and accepted_prunes >= args.max_prune_steps:
            print(f"Reached max accepted prune steps: {args.max_prune_steps}")
            break

        candidate_features = [name for name in active_features if name != feature]
        result = evaluate_feature_set(
            step_name=f"prune:{feature}",
            feature_names=candidate_features,
            X_train_full=X_train_full,
            y_train=y_train,
            X_test_full=X_test_full,
            best_params=best_params,
        )
        delta = result["best_cv"] - best_result["best_cv"]
        improved = delta > args.min_improve

        experiment_log.append(
            {
                "stage": "prune",
                "status": "accepted" if improved else "rejected_stop",
                "feature": feature,
                "feature_count": result["feature_count"],
                "best_cv": result["best_cv"],
                "best_ensemble": result["best_ensemble"],
                "delta_vs_best": delta,
            }
        )

        if improved:
            active_features = candidate_features
            best_result = result
            removed_features.append(feature)
            accepted_prunes += 1
            print(f"accepted prune: {feature} / new best CV={best_result['best_cv']:.5f}")
            continue

        prune_stop_feature = feature
        print(f"stopping prune stage at feature: {feature} / delta={delta:.5f}")
        break

    if prune_stop_feature is None:
        print("prune stage ended without a hard stop feature.")

    if args.allow_adding_unremoved_core:
        add_candidates = [
            feature for feature in CORE_DERIVED_CANDIDATES if feature in derived_features and feature not in active_features
        ]
    else:
        removed_set = set(removed_features)
        add_candidates = [
            feature
            for feature in CORE_DERIVED_CANDIDATES
            if feature in removed_set and feature in derived_features and feature not in active_features
        ]

    added_features: List[str] = []
    for add_idx, feature in enumerate(add_candidates, start=1):
        if args.max_add_steps is not None and add_idx > args.max_add_steps:
            print(f"Reached max add attempts: {args.max_add_steps}")
            break

        candidate_features = active_features + [feature]
        result = evaluate_feature_set(
            step_name=f"add:{feature}",
            feature_names=candidate_features,
            X_train_full=X_train_full,
            y_train=y_train,
            X_test_full=X_test_full,
            best_params=best_params,
        )
        delta = result["best_cv"] - best_result["best_cv"]
        improved = delta > args.min_improve

        experiment_log.append(
            {
                "stage": "add_core",
                "status": "accepted" if improved else "rejected",
                "feature": feature,
                "feature_count": result["feature_count"],
                "best_cv": result["best_cv"],
                "best_ensemble": result["best_ensemble"],
                "delta_vs_best": delta,
            }
        )

        if improved:
            active_features = candidate_features
            best_result = result
            added_features.append(feature)
            print(f"accepted add: {feature} / new best CV={best_result['best_cv']:.5f}")
        else:
            print(f"rejected add: {feature} / delta={delta:.5f}")

    log_df = pd.DataFrame(experiment_log)
    log_df.to_csv(args.output_dir / "diet_search_log.csv", index=False, encoding="utf-8-sig")

    save_feature_list(args.output_dir / "best_features.csv", best_result["feature_names"])
    best_result["importance"].to_csv(
        args.output_dir / "best_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_result["ensemble_scores"].to_csv(
        args.output_dir / "best_ensemble_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_submission = sub.copy()
    final_submission["probability"] = best_result["best_pred"].values
    final_submission.to_csv(
        args.output_dir / "best_submission.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "use_gpu": args.use_gpu,
        "importance_path": str(args.importance_path),
        "baseline_cv": baseline["best_cv"],
        "best_cv": best_result["best_cv"],
        "best_ensemble": best_result["best_ensemble"],
        "baseline_feature_count": baseline["feature_count"],
        "best_feature_count": best_result["feature_count"],
        "accepted_prunes": removed_features,
        "accepted_core_adds": added_features,
        "prune_stop_feature": prune_stop_feature,
        "run_optuna_once": args.run_optuna_once,
        "optuna_trials": args.optuna_trials if args.run_optuna_once else 0,
    }
    save_json(args.output_dir / "summary.json", summary)

    print("\n" + "=" * 80)
    print("Search finished")
    print("=" * 80)
    print(f"baseline CV      : {baseline['best_cv']:.5f}")
    print(f"best CV          : {best_result['best_cv']:.5f}")
    print(f"best ensemble    : {best_result['best_ensemble']}")
    print(f"best feature cnt : {best_result['feature_count']}")
    print(f"accepted prunes  : {removed_features}")
    print(f"accepted adds    : {added_features}")
    print(f"outputs          : {args.output_dir}")


if __name__ == "__main__":
    main()
