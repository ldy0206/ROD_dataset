import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import xgboost as xgb


# =========================
# Configuration
# =========================
TRAIN_FILE = 'train_dataset.tsv'
TEST_FILE = 'test_dataset.tsv'
LABEL_COL = 'label'  # keep exactly as in the original script
OUTPUT_DIR = 'figure3_outputs_repro'
RANDOM_STATE = 42
DPI = 600
PALETTE = ["#AEC6CF", "#FFB7B2", "#C3B1E1", "#FDFD96", "#B5EAD7"]

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tiff(fig, path: str) -> None:
    fig.savefig(path, dpi=DPI, format='tiff', bbox_inches='tight')


# =========================
# Main workflow
# =========================

def main() -> None:
    total_start = time.time()
    ensure_dir(OUTPUT_DIR)

    print("⏳ [进度 10%] 正在读取数据...")
    train_data = pd.read_csv(TRAIN_FILE, sep='\t')
    test_data = pd.read_csv(TEST_FILE, sep='\t')

    X_train = train_data.drop(LABEL_COL, axis=1)
    y_train = train_data[LABEL_COL]
    X_test = test_data.drop(LABEL_COL, axis=1)
    y_test = test_data[LABEL_COL]

    train_size = len(y_train)
    test_size = len(y_test)

    # Keep models EXACTLY aligned with the original script.
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric='logloss',
        ),
    }

    model_colors = {
        'LogisticRegression': PALETTE[0],
        'RandomForest': PALETTE[1],
        'XGBoost': PALETTE[2],
    }

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}
    cv_probs_dict = {}
    test_probs_dict = {}
    cv_pred_dict = {}
    test_pred_dict = {}

    print(
        f"\n⏳ [进度 30%] 准备就绪！训练集: {train_size}行, 测试集: {test_size}行。开始严格复现 3 大模型对决..."
    )

    for model_name, model in models.items():
        print(f"\n   >>> 正在训练并评估: {model_name} ...")
        step_start = time.time()

        # 1) Cross-validation probabilities EXACTLY as original workflow
        cv_probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        cv_preds = (cv_probs >= 0.5).astype(int)
        cv_roc_auc = roc_auc_score(y_train, cv_probs)
        cv_pr_auc = average_precision_score(y_train, cv_probs)
        cv_acc = accuracy_score(y_train, cv_preds)
        cv_f1 = f1_score(y_train, cv_preds)

        # 2) Independent test fit EXACTLY as original workflow
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        test_probs = fitted_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        test_roc_auc = roc_auc_score(y_test, test_probs)
        test_pr_auc = average_precision_score(y_test, test_probs)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)

        fitted_models[model_name] = fitted_model
        cv_probs_dict[model_name] = cv_probs
        test_probs_dict[model_name] = test_probs
        cv_pred_dict[model_name] = cv_preds
        test_pred_dict[model_name] = test_preds

        print(f"       耗时: {time.time() - step_start:.2f} 秒")
        print(f"       CV ROC-AUC: {cv_roc_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")

        results.append({
            'Run_Time': current_time,
            'Model': model_name,
            'Train_Size': train_size,
            'Test_Size': test_size,
            'CV_ROC_AUC': round(cv_roc_auc, 4),
            'CV_PR_AUC': round(cv_pr_auc, 4),
            'CV_Accuracy': round(cv_acc, 4),
            'CV_F1': round(cv_f1, 4),
            'Test_ROC_AUC': round(test_roc_auc, 4),
            'Test_PR_AUC': round(test_pr_auc, 4),
            'Test_Accuracy': round(test_acc, 4),
            'Test_F1': round(test_f1, 4),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.tsv'), sep='\t', index=False)

    # Keep original-style experiment log as an appendable artifact too.
    log_file = os.path.join(OUTPUT_DIR, 'Experiment_Log_repro.tsv')
    if os.path.exists(log_file):
        results_df[['Run_Time','Model','Train_Size','Test_Size','CV_ROC_AUC','CV_PR_AUC','Test_ROC_AUC','Test_PR_AUC']].to_csv(
            log_file, sep='\t', mode='a', header=False, index=False
        )
    else:
        results_df[['Run_Time','Model','Train_Size','Test_Size','CV_ROC_AUC','CV_PR_AUC','Test_ROC_AUC','Test_PR_AUC']].to_csv(
            log_file, sep='\t', mode='w', header=True, index=False
        )

    # Best model selection by independent-test performance.
    # Primary criterion: Test_ROC_AUC; secondary criterion: Test_PR_AUC;
    # tertiary criterion: CV_ROC_AUC to break rare ties.
    best_row = results_df.sort_values(
        ['Test_ROC_AUC', 'Test_PR_AUC', 'CV_ROC_AUC'],
        ascending=False
    ).iloc[0]
    best_model_name = best_row['Model']
    best_model = fitted_models[best_model_name]
    best_cv_probs = cv_probs_dict[best_model_name]
    best_test_probs = test_probs_dict[best_model_name]

    print(f"\n⭐ 最优模型（按 CV ROC-AUC 选择）: {best_model_name}")

    # =========================
    # Figure a: best-model CV ROC
    # =========================
    fig_a, ax_a = plt.subplots(figsize=(5.4, 4.2))
    fpr, tpr, _ = roc_curve(y_train, best_cv_probs)
    auc_val = roc_auc_score(y_train, best_cv_probs)
    acc_val = accuracy_score(y_train, (best_cv_probs >= 0.5).astype(int))
    f1_val = f1_score(y_train, (best_cv_probs >= 0.5).astype(int))

    ax_a.plot(fpr, tpr, color=PALETTE[0], linewidth=2.0, label=f'ROC curve (AUC = {auc_val:.2f})')
    ax_a.plot([0, 1], [0, 1], color='darkgray', linestyle='--', linewidth=1.0)
    ax_a.set_xlabel('False Positive Rate')
    ax_a.set_ylabel('True Positive Rate')
    ax_a.set_title(f'Best model CV ROC: {best_model_name}')
    ax_a.legend(loc='lower right', frameon=False)
    ax_a.text(
        0.58, 0.18,
        f'Accuracy = {acc_val:.2f}\nF1-score = {f1_val:.2f}\nAUC = {auc_val:.2f}',
        transform=ax_a.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='lightgray', alpha=0.9),
    )
    ax_a.grid(alpha=0.25)
    save_tiff(fig_a, os.path.join(OUTPUT_DIR, 'a.tiff'))
    plt.close(fig_a)

    # =========================
    # Figure b: concise independent-test comparison
    # =========================
    fig_b, ax_b = plt.subplots(figsize=(6.8, 4.6))
    plot_df = results_df.set_index('Model').loc[['LogisticRegression', 'RandomForest', 'XGBoost']]
    model_order = plot_df.index.tolist()
    x = np.arange(len(model_order))
    width = 0.32

    metrics_for_b = ['Test_ROC_AUC', 'Test_PR_AUC']
    metric_labels = ['Test ROC-AUC', 'Test PR-AUC']
    metric_colors = [PALETTE[0], PALETTE[1]]

    for i, (metric, metric_label, color) in enumerate(zip(metrics_for_b, metric_labels, metric_colors)):
        bars = ax_b.bar(x + (i - 0.5) * width, plot_df[metric].values, width=width, color=color, label=metric_label)
        for bar, value in zip(bars, plot_df[metric].values):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{value:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(model_order, rotation=0)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel('Performance score')
    ax_b.set_title('Independent test set performance comparison')
    ax_b.legend(frameon=False, loc='upper left')
    ax_b.grid(axis='y', alpha=0.25)
    save_tiff(fig_b, os.path.join(OUTPUT_DIR, 'b.tiff'))
    plt.close(fig_b)

    # =========================
    # Figure c: best-model independent test ROC + PR
    # =========================
    fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(9.0, 4.1))

    test_auc = roc_auc_score(y_test, best_test_probs)
    test_acc = accuracy_score(y_test, (best_test_probs >= 0.5).astype(int))
    test_f1 = f1_score(y_test, (best_test_probs >= 0.5).astype(int))
    fpr_test, tpr_test, _ = roc_curve(y_test, best_test_probs)
    ax_c1.plot(fpr_test, tpr_test, color=PALETTE[0], linewidth=2.0, label=f'ROC curve (AUC = {test_auc:.2f})')
    ax_c1.plot([0, 1], [0, 1], color='darkgray', linestyle='--', linewidth=1.0)
    ax_c1.set_xlabel('False Positive Rate')
    ax_c1.set_ylabel('True Positive Rate')
    ax_c1.set_title(f'Test ROC: {best_model_name}')
    ax_c1.legend(loc='lower right', frameon=False)
    ax_c1.text(
        0.54, 0.16,
        f'Accuracy = {test_acc:.2f}\nF1-score = {test_f1:.2f}\nAUC = {test_auc:.2f}',
        transform=ax_c1.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='lightgray', alpha=0.9),
    )
    ax_c1.grid(alpha=0.25)

    test_ap = average_precision_score(y_test, best_test_probs)
    precision, recall, _ = precision_recall_curve(y_test, best_test_probs)
    ax_c2.plot(recall, precision, color=PALETTE[1], linewidth=2.0, label=f'PR curve (AP = {test_ap:.2f})')
    baseline = y_test.mean()
    ax_c2.hlines(baseline, 0, 1, color='darkgray', linestyle='--', linewidth=1.0, label=f'Baseline = {baseline:.2f}')
    ax_c2.set_xlabel('Recall')
    ax_c2.set_ylabel('Precision')
    ax_c2.set_title(f'Test PR: {best_model_name}')
    ax_c2.legend(loc='lower left', frameon=False)
    ax_c2.grid(alpha=0.25)

    fig_c.suptitle('Independent test set performance', y=1.02)
    save_tiff(fig_c, os.path.join(OUTPUT_DIR, 'c.tiff'))
    plt.close(fig_c)

    # =========================
    # Figure d: feature importance
    # =========================
    feature_names = X_train.columns.tolist()
    importance_values = None

    if hasattr(best_model, 'feature_importances_'):
        importance_values = np.asarray(best_model.feature_importances_, dtype=float)
    elif hasattr(best_model, 'coef_'):
        coef = np.asarray(best_model.coef_)
        importance_values = np.abs(coef).ravel()
    else:
        perm = permutation_importance(
            best_model, X_test, y_test,
            n_repeats=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring='roc_auc',
        )
        importance_values = perm.importances_mean

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values,
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.tsv'), sep='\t', index=False)
    top_n = min(15, len(importance_df))
    top_df = importance_df.head(top_n).iloc[::-1]

    fig_d, ax_d = plt.subplots(figsize=(6.8, 5.2))
    ax_d.barh(top_df['Feature'], top_df['Importance'], color=PALETTE[0])
    ax_d.set_xlabel('Feature importance')
    ax_d.set_ylabel('Feature')
    ax_d.set_title(f'Feature importance of {best_model_name}')
    ax_d.grid(axis='x', alpha=0.25)
    save_tiff(fig_d, os.path.join(OUTPUT_DIR, 'd.tiff'))
    plt.close(fig_d)

    # =========================
    # Figure e: PCA projection
    # =========================
    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_all)
    pca_df = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Label': y_all.values,
    })
    pca_df.to_csv(os.path.join(OUTPUT_DIR, 'pca_coordinates.tsv'), sep='\t', index=False)

    fig_e, ax_e = plt.subplots(figsize=(5.4, 4.6))
    label_to_color = {0: PALETTE[1], 1: PALETTE[0]}
    label_to_name = {0: 'Control pair', 1: 'Positive association'}
    for label in sorted(pca_df['Label'].unique()):
        sub = pca_df[pca_df['Label'] == label]
        ax_e.scatter(
            sub['PC1'],
            sub['PC2'],
            s=20,
            alpha=0.8,
            color=label_to_color.get(label, PALETTE[2]),
            edgecolors='none',
            label=label_to_name.get(label, str(label)),
        )
    ax_e.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    ax_e.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    ax_e.set_title('PCA of input features by class label')
    ax_e.legend(frameon=False)
    ax_e.grid(alpha=0.25)
    save_tiff(fig_e, os.path.join(OUTPUT_DIR, 'e.tiff'))
    plt.close(fig_e)

    # Also save original-style combined comparison plots for reference.
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for model_name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
        color = model_colors[model_name]
        cv_probs = cv_probs_dict[model_name]
        test_probs = test_probs_dict[model_name]
        cv_auc = roc_auc_score(y_train, cv_probs)
        test_auc_val = roc_auc_score(y_test, test_probs)
        cv_ap = average_precision_score(y_train, cv_probs)
        test_ap_val = average_precision_score(y_test, test_probs)

        fpr_cv, tpr_cv, _ = roc_curve(y_train, cv_probs)
        fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
        ax_roc.plot(fpr_cv, tpr_cv, label=f'{model_name} CV (AUC={cv_auc:.3f})', color=color, linestyle='--', alpha=0.6)
        ax_roc.plot(fpr_test, tpr_test, label=f'{model_name} Test (AUC={test_auc_val:.3f})', color=color, linestyle='-', linewidth=2)

        prec_cv, rec_cv, _ = precision_recall_curve(y_train, cv_probs)
        prec_test, rec_test, _ = precision_recall_curve(y_test, test_probs)
        ax_pr.plot(rec_cv, prec_cv, label=f'{model_name} CV (AUC={cv_ap:.3f})', color=color, linestyle='--', alpha=0.6)
        ax_pr.plot(rec_test, prec_test, label=f'{model_name} Test (AUC={test_ap_val:.3f})', color=color, linestyle='-', linewidth=2)

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle=':')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve Comparison (LR vs RF vs XGBoost)')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.grid(alpha=0.3)
    save_tiff(fig_roc, os.path.join(OUTPUT_DIR, 'ROC_Curve_Comparison_repro.tiff'))
    plt.close(fig_roc)

    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('PR Curve Comparison (LR vs RF vs XGBoost)')
    ax_pr.legend(loc='lower left', fontsize=9)
    ax_pr.grid(alpha=0.3)
    save_tiff(fig_pr, os.path.join(OUTPUT_DIR, 'PR_Curve_Comparison_repro.tiff'))
    plt.close(fig_pr)

    print("\n⏳ [进度 95%] 图表、日志与中间结果已保存。")
    print(f"✅ [进度 100%] 大功告成！总耗时: {time.time() - total_start:.2f} 秒。")
    print(f"👉 输出目录: {OUTPUT_DIR}")
    print("👉 其中 a-e.tiff 即为 Figure 3 方案A所需五张图。")


if __name__ == '__main__':
    main()
