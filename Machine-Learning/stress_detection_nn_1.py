import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras
from keras import layers, callbacks

tf.config.set_visible_devices([], 'GPU')
tf.random.set_seed(42)
np.random.seed(42)


data_folder = "data"
fps = 30

offset = True
offset_s = 0.0

window_s = 10.0
stride_s = 2.0

hr_only        = False
hr_baseline_window  = 15
hr_threshold      = 9
eda_baseline_window = 15
eda_threshold         = 0.007
eda_min_var           = 1e-6

model_type = "lstm" # "lstm" or "mlp"

epochs     = 100
batch_size = 32
patience   = 50

random_holdout_seed = 42


def compute_offset(pts, eda, hr):
    if not offset:
        return offset_s
    physio_duration = max(eda["t"].max(), hr["t"].max())
    video_duration  = pts["frame"].max() / fps
    video_offset          = physio_duration - video_duration
    if video_offset < 0:
        print(f"  WARNING: Negative offset ({video_offset:.2f}s), clamping to 0.")
        video_offset = 0.0
    return video_offset


def load_trial(points_path, eda_path, hr_path):
    pts = pd.read_csv(points_path)
    eda = pd.read_csv(eda_path).rename(columns={"time_seconds": "t", "clean_value": "eda"})
    hr  = pd.read_csv(hr_path).rename(columns={"time_seconds": "t", "clean_value": "hr"})

    eda = eda.sort_values("t").reset_index(drop=True)
    hr  = hr.sort_values("t").reset_index(drop=True)

    offset   = compute_offset(pts, eda, hr)
    pts["t"] = (pts["frame"] / fps) + offset
    print(f"    Offset: {offset:.2f}s  "
          f"(physio={max(eda['t'].max(), hr['t'].max()):.1f}s, "
          f"video={pts['frame'].max()/fps:.1f}s)")

    return pts.sort_values("t").reset_index(drop=True), eda, hr


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
            print("    WARNING: EDA flat -- using HR only.")
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
    print(f"    Combined stressed: {stressed.mean()*100:.1f}%")
    return pd.DataFrame({"t": times, "stressed": stressed.astype(int)})


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



FEATURE_COLS = (
    [f"change {i}.x" for i in range(1, 7)] +
    [f"change {i}.y" for i in range(1, 7)] +
    [f"change {i}.z" for i in range(1, 7)]
)



def extract_windows_for_model(pts, labels, model_type):
    t_min = pts["t"].min()
    t_max = pts["t"].max()

    available_cols = [c for c in FEATURE_COLS if c in pts.columns]
    if not available_cols:
        return np.array([]), np.array([])

    seq_len = int(fps * window_s)

    windows_X, windows_y = [], []
    t = t_min

    while t + window_s <= t_max:
        mask   = (pts["t"] >= t) & (pts["t"] < t + window_s)
        window = pts[mask]

        if len(window) < 3:
            t += stride_s
            continue

        lmask      = (labels["t"] >= t) & (labels["t"] < t + window_s)
        win_labels = labels.loc[lmask, "stressed"].values
        if len(win_labels) == 0:
            t += stride_s
            continue
        label = int(np.mean(win_labels) >= 0.5)

        if model_type == "lstm":
            frame_data = window[available_cols].values.astype(np.float32)
            if len(frame_data) < 2:
                t += stride_s
                continue
            old_idx = np.linspace(0, 1, len(frame_data))
            new_idx = np.linspace(0, 1, seq_len)
            resampled = np.stack(
                [np.interp(new_idx, old_idx, frame_data[:, col_i])
                 for col_i in range(frame_data.shape[1])],
                axis=1
            )
            windows_X.append(resampled)

        else:
            feats = []
            for col in available_cols:
                vals = window[col].values.astype(np.float32)
                feats.extend([vals.mean(), vals.std(), vals.max(),
                               vals.min(), np.abs(np.diff(vals)).mean()])
            windows_X.append(np.array(feats, dtype=np.float32))

        windows_y.append(label)
        t += stride_s

    if not windows_X:
        return np.array([]), np.array([])

    return np.stack(windows_X), np.array(windows_y)



def normalize_sequences(X_train, X_test):
    shape = X_train.shape
    flat_train = X_train.reshape(-1, shape[-1])
    flat_test  = X_test.reshape(-1, shape[-1])

    scaler = StandardScaler()
    flat_train = scaler.fit_transform(flat_train)
    flat_test  = scaler.transform(flat_test)

    return (flat_train.reshape(shape),
            flat_test.reshape(X_test.shape[0], *shape[1:]),
            scaler)



def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="stress_mlp")
    return model


def build_lstm(seq_len, n_features):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="stress_lstm")
    return model



def build_full_dataset(data_root, model_type):
    trials = discover_trials(data_root)
    if not trials:
        raise FileNotFoundError(
            f"No trial data found under '{data_root}'. "
            "Check DATA_ROOT and your directory structure."
        )
    print(f"Found {len(trials)} trials across: "
          f"{sorted(set(t[0] for t in trials))}\n")

    all_X, all_y, all_participants, all_trials = [], [], [], []

    for participant_id, trial_id, pts_path, eda_path, hr_path in trials:
        print(f"  Processing {participant_id}/{trial_id}...")
        try:
            pts, eda, hr = load_trial(pts_path, eda_path, hr_path)
            labels       = generate_stress_label_series(eda, hr)
            X, y         = extract_windows_for_model(pts, labels, model_type)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        if len(y) == 0:
            print(f"    Skipped (no valid windows).")
            continue

        print(f"    Windows: {len(y)}  |  "
              f"stressed: {y.sum()} ({100*y.mean():.1f}%)  |  "
              f"unstressed: {(y==0).sum()} ({100*(1-y.mean()):.1f}%)\n")

        all_X.append(X)
        all_y.append(y)
        all_participants.extend([participant_id] * len(y))
        all_trials.extend([trial_id] * len(y))

    if not all_X:
        raise RuntimeError("No usable windows extracted. Check FPS and file paths.")

    return np.concatenate(all_X), np.concatenate(all_y), np.array(all_participants), np.array(all_trials)



def train_nn(X_train, y_train, model_type, seq_len, n_features):
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {
        0: (n_neg + n_pos) / (2.0 * n_neg) if n_neg > 0 else 1.0,
        1: (n_neg + n_pos) / (2.0 * n_pos) if n_pos > 0 else 1.0,
    }
    model = build_lstm(seq_len, n_features) if model_type == "lstm" else build_mlp(n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=[callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0
        )],
        verbose=1
    )
    return model


def report_nn_results(all_true, all_preds, per_participant, label):
    print(f"\n-- {label} --")
    print(classification_report(all_true, all_preds, target_names=["unstressed", "stressed"]))
    valid_aucs = [v["auc"] for v in per_participant.values() if not np.isnan(v["auc"])]
    valid_f1s  = [v["f1"]  for v in per_participant.values()]
    if valid_aucs:
        print(f"Mean AUC: {np.mean(valid_aucs):.3f} (+/-{np.std(valid_aucs):.3f})")
    print(f"Mean F1:  {np.mean(valid_f1s):.3f} (+/-{np.std(valid_f1s):.3f})")


def select_random_holdout_trials(participants, trials, seed=None):
    rng = np.random.default_rng(seed)
    holdout_index = np.zeros(len(participants), dtype=bool)

    for pid in np.unique(participants):
        pid_mask         = participants == pid
        available_trials = np.unique(trials[pid_mask])
        if len(available_trials) < 2:
            print(f"  WARNING: {pid} has only 1 trial — excluding from holdout set.")
            continue
        chosen = rng.choice(available_trials)
        holdout_index[pid_mask & (trials == chosen)] = True
        print(f"  {pid}: holding out trial '{chosen}' "
              f"(options were {sorted(available_trials)})")

    return holdout_index


def trial_holdout_evaluate(X, y, participants, trials, model_type):
    seq_len    = X.shape[1] if model_type == "lstm" else None
    n_features = X.shape[2] if model_type == "lstm" else X.shape[1]

    print("  Randomly selecting one holdout trial per participant...")
    test_mask  = select_random_holdout_trials(participants, trials, seed=random_holdout_seed)
    train_mask = ~test_mask

    if train_mask.sum() == 0:
        raise ValueError("No training data — all windows were selected as holdout.")
    if test_mask.sum() == 0:
        raise ValueError("No test windows selected. Check that participants have >1 trial.")

    n_test_p = len(np.unique(participants[test_mask]))
    print(f"\n  Holding out randomly chosen trials from {n_test_p} participants "
          f"({test_mask.sum()} test windows).")
    print(f"  Training on {train_mask.sum()} windows from all remaining trials...\n")

    X_train, y_train = X[train_mask], y[train_mask]
    X_train, _, scaler = normalize_sequences(X_train, X_train)
    X_train = np.nan_to_num(X_train, nan=0.0)

    model = train_nn(X_train, y_train, model_type, seq_len, n_features)
    print("  Training complete. Evaluating on held-out trials...\n")

    all_true, all_preds, all_probs = [], [], []
    per_participant = {}

    for pid in sorted(np.unique(participants[test_mask])):
        mask   = participants == pid
        mask  &= test_mask
        if mask.sum() == 0:
            continue

        X_test = X[mask]
        flat   = X_test.reshape(-1, X_test.shape[-1])
        flat   = scaler.transform(flat)
        X_test = np.nan_to_num(flat.reshape(X_test.shape), nan=0.0)
        y_test = y[mask]

        probs = model.predict(X_test, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)

        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
        per_participant[pid] = {"f1": f1, "auc": auc, "n_windows": len(y_test)}

        all_true.extend(y_test)
        all_preds.extend(preds)
        all_probs.extend(probs)

        held_trial = np.unique(trials[mask])[0]
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "  nan"
        print(f"  {pid}/{held_trial}: F1={f1:.3f}, AUC={auc_str} ({len(y_test)} windows)")

    report_nn_results(all_true, all_preds, per_participant,
                       "Trial holdout results (random holdout per participant)")
    keras.backend.clear_session()
    return per_participant



def print_model_summary(model_type, X):
    if model_type == "lstm":
        model = build_lstm(X.shape[1], X.shape[2])
    else:
        model = build_mlp(X.shape[1])
    model.summary()
    keras.backend.clear_session()



if __name__ == "__main__":
    print("=" * 60)
    print(f"Stress Detection — Neural Network  |  FPS={fps}")
    print(f"Model: {model_type.upper()}  |  Window={window_s}s  Stride={stride_s}s")
    print(f"epochs={epochs}  BatchSize={batch_size}  EarlyStop patience={patience}")
    print(f"Holdout: random trial per participant  |  seed={random_holdout_seed}")
    print("=" * 60 + "\n")

    X, y, participants, trials = build_full_dataset(data_folder, model_type)

    print(f"\nTotal windows:  {len(y)}")
    print(f"Stressed:       {y.sum()} ({100*y.mean():.1f}%)")
    print(f"Unstressed:     {(y==0).sum()} ({100*(1-y.mean()):.1f}%)")
    print(f"Participants:   {len(np.unique(participants))}")
    if model_type == "lstm":
        print(f"Sequence shape: {X.shape}  (windows x frames x features)\n")
    else:
        print(f"Feature shape:  {X.shape}  (windows x features)\n")

    print("-- Model architecture ---------------------------------------")
    print_model_summary(model_type, X)

    print("\n-- Leave-One-Participant-Out evaluation ---------------------")
    trial_holdout_evaluate(X, y, participants, trials, model_type)

    print("\nDone.")