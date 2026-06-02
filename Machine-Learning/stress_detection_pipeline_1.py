import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")


data_folder = "data"
fps = 30

offset = True
offset_s = 0.0

window_s = 4.8
stride_s = 2

hr_only = False
hr_baseline_window = 15
hr_threshold = 9
eda_baseline_window = 15
eda_threshold = 0.007
eda_min_var = 1e-6

participant_normalize = True

model_type = "random_forest"   # "random_forest","gradient_boosting","svm"
n_estimators = 150
random_state = 42
random_holdout_seed = 42

def compute_offset(pts, eda, hr):
    if not offset:
        return offset_s

    physio_duration = max(eda["t"].max(), hr["t"].max())
    video_duration  = pts["frame"].max() / fps
    video_offset          = physio_duration - video_duration

    if video_offset < 0:
        print(f"  WARNING: Computed offset is negative ({video_offset:.2f}s) -- "
              "video appears longer than physio recording. Clamping to 0.")
        video_offset = 0.0

    return video_offset


def load_trial(points_path, eda_path, hr_path):
    pts = pd.read_csv(points_path)
    eda = pd.read_csv(eda_path)
    hr  = pd.read_csv(hr_path)

    eda = eda.rename(columns={"time_seconds": "t", "clean_value": "eda"})
    hr  = hr.rename(columns={"time_seconds": "t", "clean_value": "hr"})

    eda = eda.sort_values("t").reset_index(drop=True)
    hr  = hr.sort_values("t").reset_index(drop=True)

    offset = compute_offset(pts, eda, hr)
    pts["t"] = (pts["frame"] / fps) + offset
    print(f"    Offset: {offset:.2f}s  "
          f"(physio={max(eda['t'].max(), hr['t'].max()):.1f}s, "
          f"video={pts['frame'].max()/fps:.1f}s)")

    pts = pts.sort_values("t").reset_index(drop=True)
    return pts, eda, hr


def generate_stress_label_series(eda, hr):
    duration = max(eda["t"].max(), hr["t"].max())
    times    = np.arange(0, duration, 1.0)

    hr_interp     = np.interp(times, hr["t"].values, hr["hr"].values)
    baseline_mask = times <= hr_baseline_window
    hr_baseline   = hr_interp[baseline_mask].mean() if baseline_mask.sum() >= 2 else hr_interp.mean()
    hr_stressed   = (hr_interp - hr_baseline) > hr_threshold

    print(f"    HR  baseline: {hr_baseline:.1f} bpm  "
          f"threshold: >{hr_baseline + hr_threshold:.1f} bpm  "
          f"stressed: {hr_stressed.mean()*100:.1f}%")

    eda_var = eda["eda"].var()
    if hr_only or eda_var < eda_min_var:
        if not hr_only:
            print("    WARNING: EDA signal flat -- using HR only for this trial.")
        eda_stressed = np.zeros(len(times), dtype=bool)
    else:
        eda_interp    = np.interp(times, eda["t"].values, eda["eda"].values)
        baseline_mask = times <= eda_baseline_window
        eda_baseline  = eda_interp[baseline_mask].mean() if baseline_mask.sum() >= 2 else eda_interp.mean()
        eda_stressed  = (eda_interp - eda_baseline) > eda_threshold

        print(f"    EDA baseline: {eda_baseline:.4f}  "
              f"threshold: >{eda_baseline + eda_threshold:.4f}  "
              f"stressed: {eda_stressed.mean()*100:.1f}%")

    stressed = hr_stressed | eda_stressed
    print(f"    Combined stressed: {stressed.mean()*100:.1f}% of seconds")
    return pd.DataFrame({"t": times, "stressed": stressed.astype(int)})



def extract_window_features(window):
    feats  = {}
    speeds = []

    for point_id in range(1, 7):
        xcol = f"change {point_id}.x"
        ycol = f"change {point_id}.y"
        zcol = f"change {point_id}.z"

        if xcol not in window.columns:
            continue

        dx = window[xcol].values
        dy = window[ycol].values
        dz = window[zcol].values if zcol in window.columns else np.zeros(len(dx))

        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        accel = np.abs(np.diff(speed, prepend=speed[0]))

        feats[f"p{point_id}_mean_dx"]     = np.mean(dx)
        feats[f"p{point_id}_std_dx"]      = np.std(dx)
        feats[f"p{point_id}_mean_dy"]     = np.mean(dy)
        feats[f"p{point_id}_std_dy"]      = np.std(dy)
        feats[f"p{point_id}_mean_dz"]     = np.mean(dz)
        feats[f"p{point_id}_std_dz"]      = np.std(dz)
        feats[f"p{point_id}_speed_mean"]  = np.mean(speed)
        feats[f"p{point_id}_speed_std"]   = np.std(speed)
        feats[f"p{point_id}_speed_max"]   = np.max(speed)
        feats[f"p{point_id}_accel_mean"]  = np.mean(accel)
        feats[f"p{point_id}_accel_max"]   = np.max(accel)

        if len(speed) > 1:
            feats[f"p{point_id}_speed_trend"] = np.polyfit(range(len(speed)), speed, 1)[0]
        else:
            feats[f"p{point_id}_speed_trend"] = 0.0

        speeds.append(speed)

    all_delta_cols = [c for c in window.columns if c.startswith("change")]
    feats["motion_energy"] = np.sum(window[all_delta_cols].values ** 2)

    if len(speeds) >= 2:
        corrs = np.corrcoef(np.vstack(speeds))
        idx   = np.triu_indices(len(speeds), k=1)
        for r, c in zip(idx[0], idx[1]):
            feats[f"corr_p{r+1}_p{c+1}"] = corrs[r, c]

    return feats



def build_trial_feature_matrix(pts, labels):
    t_min = pts["t"].min()
    t_max = pts["t"].max()

    rows_X, rows_y = [], []
    t = t_min
    while t + window_s <= t_max:
        mask   = (pts["t"] >= t) & (pts["t"] < t + window_s)
        window = pts[mask]
        if len(window) < 3:
            t += stride_s
            continue

        feats = extract_window_features(window)

        lmask      = (labels["t"] >= t) & (labels["t"] < t + window_s)
        win_labels = labels.loc[lmask, "stressed"].values
        if len(win_labels) == 0:
            t += stride_s
            continue

        label = int(np.mean(win_labels) >= 0.5)
        rows_X.append(feats)
        rows_y.append(label)
        t += stride_s

    if not rows_X:
        return pd.DataFrame(), np.array([])

    return pd.DataFrame(rows_X), np.array(rows_y)



def discover_trials(data_folder):
    trials = []
    for pts_path in sorted(glob.glob(f"{data_folder}/**/*Points*.csv", recursive=True)):
        trial_dir = os.path.dirname(pts_path)
        eda_path  = os.path.join(trial_dir, "eda_clean.csv")
        hr_path   = os.path.join(trial_dir, "hr_clean.csv")
        if not (os.path.exists(eda_path) and os.path.exists(hr_path)):
            print(f"  WARNING: Skipping {pts_path}: missing eda or hr file.")
            continue
        parts          = pts_path.replace("\\", "/").split("/")
        participant_id = next((p for p in parts if p.startswith("P")), "UNK")
        trial_id       = next((p for p in parts if p.startswith("T")), "UNK")
        trials.append((participant_id, trial_id, pts_path, eda_path, hr_path))
    return trials



def normalize_per_participant(full_X, feature_cols):
    full_X = full_X.copy()
    for pid in full_X["participant"].unique():
        mask             = full_X["participant"] == pid
        participant_data = full_X.loc[mask, feature_cols]
        means            = participant_data.mean()
        stds             = participant_data.std().replace(0, 1)
        full_X.loc[mask, feature_cols] = (participant_data - means) / stds
    return full_X


def build_full_dataset(data_folder):
    trials = discover_trials(data_folder)
    if not trials:
        raise FileNotFoundError(
            f"No trial data found under '{data_folder}'. "
            "Check data_folder and your directory structure."
        )
    print(f"Found {len(trials)} trials across participants: "
          f"{sorted(set(t[0] for t in trials))}\n")

    all_X, all_y = [], []

    for participant_id, trial_id, pts_path, eda_path, hr_path in trials:
        print(f"  Processing {participant_id}/{trial_id}...")
        try:
            pts, eda, hr = load_trial(pts_path, eda_path, hr_path)
            labels       = generate_stress_label_series(eda, hr)
            X, y         = build_trial_feature_matrix(pts, labels)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if len(y) == 0:
            print(f"    Skipped (no valid windows).")
            continue

        n_stressed = y.sum()
        print(f"    Windows: {len(y)}  |  "
              f"stressed: {n_stressed} ({100*y.mean():.1f}%)  |  "
              f"unstressed: {len(y)-n_stressed} ({100*(1-y.mean()):.1f}%)\n")

        X["participant"] = participant_id
        X["trial"]       = trial_id
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise RuntimeError("No usable windows were extracted. Check FPS and file paths.")

    full_X = pd.concat(all_X, ignore_index=True)
    full_y = np.concatenate(all_y)

    if participant_normalize:
        feature_cols = [c for c in full_X.columns if c not in ("participant", "trial")]
        print("Applying per-participant feature normalisation...")
        full_X = normalize_per_participant(full_X, feature_cols)

    return full_X, full_y


def build_model():
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=4, random_state=random_state
        )
    elif model_type == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=random_state)
    else:
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        )


def select_random_holdout_trials(full_X, seed=None):
    rng = np.random.default_rng(seed)
    holdout_index = np.zeros(len(full_X), dtype=bool)

    for pid in full_X["participant"].unique():
        pid_mask         = full_X["participant"] == pid
        available_trials = full_X.loc[pid_mask, "trial"].unique()
        if len(available_trials) < 2:
            print(f"  WARNING: {pid} has only 1 trial — excluding from holdout set.")
            continue
        chosen = rng.choice(available_trials)
        holdout_index[pid_mask & (full_X["trial"] == chosen)] = True
        print(f"  {pid}: holding out trial '{chosen}' "
              f"(options were {sorted(available_trials)})")

    return holdout_index


def trial_holdout_evaluate(full_X, full_y):
    feature_cols = [c for c in full_X.columns if c not in ("participant", "trial")]

    print("  Randomly selecting one holdout trial per participant...")
    test_mask  = select_random_holdout_trials(full_X, seed=random_holdout_seed)
    train_mask = ~test_mask

    if train_mask.sum() == 0:
        raise ValueError("No training data — all windows were selected as holdout.")
    if test_mask.sum() == 0:
        raise ValueError("No test windows selected. Check that participants have >1 trial.")

    n_test_p = full_X.loc[test_mask, "participant"].nunique()
    print(f"\n  Holding out randomly chosen trials from {n_test_p} participants "
          f"({test_mask.sum()} test windows).")
    print(f"  Training on {train_mask.sum()} windows from all remaining trials...\n")

    X_train = full_X.loc[train_mask, feature_cols].values
    y_train = full_y[train_mask]
    X_test  = full_X.loc[test_mask,  feature_cols].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    model = build_model()
    model.fit(X_train, y_train)
    print("  Training complete. Evaluating on held-out trials...\n")

    test_indices = np.where(test_mask)[0]
    test_pos     = {orig_i: pos_i for pos_i, orig_i in enumerate(test_indices)}

    all_preds, all_true, all_probs = [], [], []
    per_participant = {}

    for pid in sorted(full_X.loc[test_mask, "participant"].unique()):
        pid_test_orig = full_X.index[(full_X["participant"] == pid) & test_mask].tolist()
        if not pid_test_orig:
            continue

        pos  = [test_pos[i] for i in pid_test_orig]
        X_p  = X_test[pos]
        y_test = full_y[pid_test_orig]

        preds = model.predict(X_p)
        probs = model.predict_proba(X_p)[:, 1] if hasattr(model, "predict_proba") else preds

        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
        per_participant[pid] = {"f1": f1, "auc": auc, "n_windows": len(y_test)}

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_probs.extend(probs)

        held_trial = full_X.loc[pid_test_orig[0], "trial"]
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "  nan"
        print(f"  {pid}/{held_trial}: F1={f1:.3f}, AUC={auc_str} ({len(y_test)} windows)")

    print("\n-- Trial holdout results (random holdout per participant) ---")
    print(classification_report(all_true, all_preds, target_names=["unstressed", "stressed"]))

    valid_aucs = [v["auc"] for v in per_participant.values() if not np.isnan(v["auc"])]
    valid_f1s  = [v["f1"]  for v in per_participant.values()]
    if valid_aucs:
        print(f"Mean AUC across participants: {np.mean(valid_aucs):.3f} "
              f"(+/-{np.std(valid_aucs):.3f})")
    print(f"Mean F1  across participants: {np.mean(valid_f1s):.3f} "
          f"(+/-{np.std(valid_f1s):.3f})")

    return per_participant, all_true, all_preds, all_probs


def show_feature_importance(full_X, full_y, top_n=20):
    feature_cols = [c for c in full_X.columns if c not in ("participant", "trial")]
    X = StandardScaler().fit_transform(
        np.nan_to_num(full_X[feature_cols].values, nan=0.0)
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        random_state=random_state, n_jobs=-1
    )
    model.fit(X, full_y)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print(f"\n-- Top {top_n} features -------------------------------------")
    print(importances.nlargest(top_n).to_string())
    return importances


if __name__ == "__main__":
    print("=" * 60)
    print(f"Stress Detection Pipeline  |  FPS={fps}")
    print(f"Window={window_s}s  Stride={stride_s}s  Model={model_type}")
    print(f"HR threshold: +{hr_threshold} bpm  |  "
          f"HR only labels: {hr_only}  |  "
          f"Per-participant normalisation: {participant_normalize}")
    print("=" * 60 + "\n")

    full_X, full_y = build_full_dataset(data_folder)

    print(f"\nTotal windows: {len(full_y)}")
    print(f"Stressed:   {full_y.sum()} ({100*full_y.mean():.1f}%)")
    print(f"Unstressed: {(full_y==0).sum()} ({100*(1-full_y.mean()):.1f}%)")
    print(f"Participants: {full_X['participant'].nunique()}\n")

    print("-- Trial holdout evaluation ---------------------------------")
    per_participant, all_true, all_preds, all_probs = trial_holdout_evaluate(full_X, full_y)

    show_feature_importance(full_X, full_y)

    print("\nDone.")
