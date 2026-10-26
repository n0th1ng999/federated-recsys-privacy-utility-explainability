import gc
import copy
import os
from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

GRAD_CLIP_NORM = float(os.getenv("RECSYS_GRAD_CLIP_NORM", "5.0"))
GC_EVERY_EPOCH = os.getenv("RECSYS_GC_EVERY_EPOCH", "0").strip().lower() in {"1", "true", "yes"}
EMPTY_CUDA_CACHE_EVERY_EPOCH = os.getenv("RECSYS_EMPTY_CUDA_CACHE_EVERY_EPOCH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


def _configure_torch_backends():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")


def clean_nan(preds, labels):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    mask = ~np.isnan(preds) & ~np.isnan(labels)
    return preds[mask], labels[mask]


def _tensor_nonfinite_summary(tensor):
    tensor = tensor.detach()
    nan_count = int(torch.isnan(tensor).sum().item()) if tensor.is_floating_point() else 0
    inf_count = int(torch.isinf(tensor).sum().item()) if tensor.is_floating_point() else 0
    parts = [f"shape={tuple(tensor.shape)}", f"nan={nan_count}", f"inf={inf_count}"]
    if tensor.numel() > 0 and tensor.is_floating_point():
        safe = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(f"min={float(safe.min().item()):.6f}")
        parts.append(f"max={float(safe.max().item()):.6f}")
    return ", ".join(parts)


def _ensure_finite_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        return
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(f"Non-finite tensor detected in {name}: {_tensor_nonfinite_summary(tensor)}")


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=(device.type == "cuda"))
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    return value


def _flatten_tensor_shapes(prefix, value, out):
    if isinstance(value, torch.Tensor):
        out[prefix] = tuple(value.shape)
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_tensor_shapes(next_prefix, sub, out)


def _ensure_finite_nested(prefix, value):
    if isinstance(value, torch.Tensor):
        _ensure_finite_tensor(prefix, value)
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _ensure_finite_nested(next_prefix, sub)


def expected_calibration_error(y_true, y_pred, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        avg_conf = y_pred[mask].mean()
        avg_true = y_true[mask].mean()
        ece += np.abs(avg_conf - avg_true) * mask.mean()
    return float(ece)


def _build_inputs(batch, model, strict_branches=False, required_branches=None):
    branch_order = getattr(model, "branch_order", None)
    if not branch_order:
        branches = getattr(model, "branches", None)
        if branches is not None and hasattr(branches, "keys"):
            branch_order = list(branches.keys())
    if not branch_order:
        branch_order = ["user_state", "collaborative", "pair_context"]

    required = branch_order if required_branches is None else list(required_branches)
    if strict_branches:
        missing = [k for k in required if k not in batch]
        if missing:
            raise RuntimeError(f"Missing required branch tensors in batch: {missing}")

    inputs = {k: batch[k] for k in branch_order if k in batch}
    for key in ("pair_context",):
        if key in batch:
            inputs[key] = batch[key]
    return inputs


def _derive_conf_target(reg_logits, rank_logits, labels_reg, labels_rank):
    reg_prob = torch.sigmoid(reg_logits.detach()).clamp(0.0, 1.0)
    rank_prob = torch.sigmoid(rank_logits.detach()).clamp(0.0, 1.0)
    rank_target = ((labels_rank - 1.0) / 4.0).clamp(0.0, 1.0)

    reg_err = torch.abs(reg_prob - labels_reg)
    rank_err = torch.abs(rank_prob - rank_target)
    combined_err = 0.65 * reg_err + 0.35 * rank_err

    # Confidence should stay high when the model correctly lands in any of the
    # four reg/rank quadrants, including valid low-buy/high-rank cases.
    pred_gap = reg_prob - rank_prob
    target_gap = labels_reg - rank_target
    gap_err = torch.abs(pred_gap - target_gap)

    conf_target = torch.clamp(1.0 - (0.70 * combined_err + 0.30 * gap_err), 0.0, 1.0)
    return conf_target


def pairwise_rank_loss(scores, labels_rank, user_idx):
    """
    User-wise pairwise ranking:
    for same user, enforce score(i) > score(j) when label_rank(i) > label_rank(j).
    """
    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    if user_idx.numel() <= 1:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    group_starts_mask = torch.ones_like(user_idx, dtype=torch.bool)
    group_starts_mask[1:] = user_idx[1:] != user_idx[:-1]
    grouped_user_ids = user_idx[group_starts_mask]
    if int(torch.unique(grouped_user_ids).numel()) == int(grouped_user_ids.numel()):
        start_positions = torch.nonzero(group_starts_mask, as_tuple=False).flatten()
        end_positions = torch.empty_like(start_positions)
        if start_positions.numel() > 1:
            end_positions[:-1] = start_positions[1:]
        end_positions[-1] = int(user_idx.numel())

        losses = []
        for start, end in zip(start_positions.tolist(), end_positions.tolist()):
            if int(end - start) < 2:
                continue
            s = scores[start:end]
            y = labels_rank[start:end]
            pref = y.unsqueeze(1) - y.unsqueeze(0)
            pos_pairs = pref > 1e-6
            if not torch.any(pos_pairs):
                continue
            score_diff = s.unsqueeze(1) - s.unsqueeze(0)
            losses.append(F.softplus(-score_diff[pos_pairs]).mean())

        if not losses:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return torch.stack(losses).mean()

    losses = []
    for uid in torch.unique(user_idx):
        mask = user_idx == uid
        if int(mask.sum().item()) < 2:
            continue

        s = scores[mask]
        y = labels_rank[mask]

        pref = y.unsqueeze(1) - y.unsqueeze(0)
        pos_pairs = pref > 1e-6
        if not torch.any(pos_pairs):
            continue

        score_diff = s.unsqueeze(1) - s.unsqueeze(0)
        losses.append(F.softplus(-score_diff[pos_pairs]).mean())

    if not losses:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    return torch.stack(losses).mean()


def _compute_reg_metrics(preds_reg_all, labels_reg_all):
    if not preds_reg_all or not labels_reg_all:
        return {
            "mae_reg": float("nan"),
            "rmse_reg": float("nan"),
            "brier_reg": float("nan"),
            "ece_reg": float("nan"),
        }

    preds_reg = torch.cat(preds_reg_all).numpy()
    labels_reg = torch.cat(labels_reg_all).numpy()
    preds_reg, labels_reg = clean_nan(preds_reg, labels_reg)

    if preds_reg.size == 0:
        return {
            "mae_reg": float("nan"),
            "rmse_reg": float("nan"),
            "brier_reg": float("nan"),
            "ece_reg": float("nan"),
        }

    return {
        "mae_reg": mean_absolute_error(labels_reg, preds_reg),
        "rmse_reg": np.sqrt(mean_squared_error(labels_reg, preds_reg)),
        "brier_reg": np.mean((preds_reg - labels_reg) ** 2),
        "ece_reg": expected_calibration_error(labels_reg, preds_reg),
    }


def _safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if y_true.size < 2:
        return float("nan")

    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))

    rt_std = float(np.std(rt))
    rp_std = float(np.std(rp))
    if rt_std <= 1e-12 or rp_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rt, rp)[0, 1])


def _safe_ndcg_at_k(user_idx_all, labels_rank_all, preds_rank_all, k=10):
    if not user_idx_all or not labels_rank_all or not preds_rank_all:
        return float("nan")

    users = np.concatenate(user_idx_all)
    y_true = np.concatenate(labels_rank_all)
    y_pred = np.concatenate(preds_rank_all)

    if users.size == 0:
        return float("nan")

    ndcgs = []
    for uid in np.unique(users):
        mask = users == uid
        if np.sum(mask) < 2:
            continue

        t = y_true[mask]
        p = y_pred[mask]

        order_pred = np.argsort(-p)
        order_true = np.argsort(-t)
        kk = min(k, len(t))

        rel_pred = t[order_pred[:kk]]
        rel_true = t[order_true[:kk]]

        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = np.sum((np.power(2.0, rel_pred) - 1.0) * discounts)
        idcg = np.sum((np.power(2.0, rel_true) - 1.0) * discounts)
        if idcg > 1e-12:
            ndcgs.append(float(dcg / idcg))

    if not ndcgs:
        return float("nan")
    return float(np.mean(ndcgs))


def _compute_rank_metrics(preds_rank_all, labels_rank_all, user_idx_all):
    if not preds_rank_all or not labels_rank_all:
        return {
            "mae_rank": float("nan"),
            "rmse_rank": float("nan"),
            "spearman_rank": float("nan"),
            "ndcg10_rank": float("nan"),
        }

    preds_rank = np.concatenate(preds_rank_all)
    labels_rank = np.concatenate(labels_rank_all)
    preds_rank, labels_rank = clean_nan(preds_rank, labels_rank)
    if preds_rank.size == 0:
        return {
            "mae_rank": float("nan"),
            "rmse_rank": float("nan"),
            "spearman_rank": float("nan"),
            "ndcg10_rank": float("nan"),
        }

    return {
        "mae_rank": mean_absolute_error(labels_rank, preds_rank),
        "rmse_rank": np.sqrt(mean_squared_error(labels_rank, preds_rank)),
        "spearman_rank": _safe_spearman(labels_rank, preds_rank),
        "ndcg10_rank": _safe_ndcg_at_k(user_idx_all, labels_rank_all, preds_rank_all, k=10),
    }


def _append_metric_batch(
    preds_reg_all,
    labels_reg_all,
    preds_rank_all,
    labels_rank_all,
    user_idx_all,
    *,
    reg_prob,
    labels_reg,
    rank_pred,
    labels_rank,
    user_idx,
    metrics_max_samples,
    collected_rows,
):
    if metrics_max_samples is not None and metrics_max_samples > 0:
        remaining = int(metrics_max_samples - collected_rows)
        if remaining <= 0:
            return collected_rows
        take = min(int(labels_reg.shape[0]), remaining)
    else:
        take = int(labels_reg.shape[0])

    if take <= 0:
        return collected_rows

    if take < int(labels_reg.shape[0]):
        sample_idx = torch.randperm(int(labels_reg.shape[0]), device=labels_reg.device)[:take]
        reg_prob = reg_prob.index_select(0, sample_idx)
        labels_reg = labels_reg.index_select(0, sample_idx)
        rank_pred = rank_pred.index_select(0, sample_idx)
        labels_rank = labels_rank.index_select(0, sample_idx)
        user_idx = user_idx.index_select(0, sample_idx)

    preds_reg_all.append(reg_prob.detach().cpu())
    labels_reg_all.append(labels_reg.detach().cpu())
    preds_rank_all.append(rank_pred.detach().cpu().numpy())
    labels_rank_all.append(labels_rank.detach().cpu().numpy())
    user_idx_all.append(user_idx.detach().cpu().numpy())
    return int(collected_rows + take)


def train_loop(
    model,
    train_loader,
    val_loader=None,
    num_epochs=15,
    lr=0.003,
    patience=5,
    lambda_rank=0.3,
    lambda_pairwise=0.5,
    lambda_conf=0.1,
    progress_callback=None,
    progress_interval=25,
    collect_train_metrics=True,
    validate_every=1,
    metrics_max_samples=0,
    verbose_stdout=True,
):
    _configure_torch_backends()
    device = torch.device("cuda")
    model.to(device)
    gpu_name = torch.cuda.get_device_name(device)
    if verbose_stdout:
        print(f"Training on CUDA: {gpu_name}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_reg_fn = nn.BCEWithLogitsLoss()
    loss_reg_mse_fn = nn.MSELoss()
    loss_rank_point_fn = nn.SmoothL1Loss()
    loss_conf_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0
    validate_every = max(1, int(validate_every))
    metric_limit = int(metrics_max_samples) if metrics_max_samples and metrics_max_samples > 0 else None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_reg_bce = 0.0
        total_reg_mse = 0.0
        total_rank_point = 0.0
        total_rank_pair = 0.0
        total_conf_loss = 0.0
        n_batches = 0
        preds_reg_all, labels_reg_all = [], []
        preds_rank_all, labels_rank_all, user_idx_all = [], [], []
        logged_input_shapes = False
        metric_rows = 0
        samples_seen = 0
        epoch_t0 = perf_counter()
        train_t0 = epoch_t0
        data_wait_seconds = 0.0
        host_to_device_seconds = 0.0
        compute_seconds = 0.0

        if verbose_stdout:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_train_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        callback_interval = max(1, int(progress_interval or 1))
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_started",
                    "epoch": int(epoch + 1),
                    "num_epochs": int(num_epochs),
                    "train_batches": int(total_train_batches or 0),
                    "val_batches": int(len(val_loader)) if val_loader is not None and hasattr(val_loader, "__len__") else 0,
                    "percent": float(100.0 * epoch / max(num_epochs, 1)),
                }
            )

        batch_loop_end = perf_counter()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", leave=False, disable=not verbose_stdout)):
            batch_ready = perf_counter()
            data_wait_seconds += batch_ready - batch_loop_end
            move_t0 = perf_counter()
            batch = _move_to_device(batch, device)
            host_to_device_seconds += perf_counter() - move_t0
            optimizer.zero_grad(set_to_none=True)
            compute_t0 = perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                inputs = _build_inputs(
                    batch,
                    model,
                    strict_branches=True,
                    required_branches=getattr(model, "required_branches", None),
                )
                if not logged_input_shapes and epoch == 0:
                    shape_report = {}
                    for key, value in inputs.items():
                        _flatten_tensor_shapes(key, value, shape_report)
                    if verbose_stdout:
                        print(f"Training input tensors (first batch): {shape_report}")
                    logged_input_shapes = True
                if epoch == 0 and batch_idx == 0:
                    for key, value in inputs.items():
                        _ensure_finite_nested(f"train inputs[{key}]", value)
                reg_logits, rank_logits, conf_logits = model(inputs)

                reg_logits = reg_logits.view(-1)
                rank_logits = rank_logits.view(-1)
                conf_logits = conf_logits.view(-1)
                labels_reg = batch["label_reg"].view(-1)
                labels_rank = batch["label_rank"].view(-1).clamp(1.0, 5.0)
                user_idx = batch["user_idx"].view(-1)
                if epoch == 0 and batch_idx == 0:
                    _ensure_finite_tensor("train labels_reg", labels_reg)
                    _ensure_finite_tensor("train labels_rank", labels_rank)

                _ensure_finite_tensor("train reg_logits", reg_logits)
                _ensure_finite_tensor("train rank_logits", rank_logits)
                _ensure_finite_tensor("train conf_logits", conf_logits)

                reg_prob = torch.sigmoid(reg_logits)
                loss_reg_bce = loss_reg_fn(reg_logits, labels_reg)
                loss_reg_mse = loss_reg_mse_fn(reg_prob, labels_reg)
                loss_reg = 0.7 * loss_reg_bce + 0.3 * loss_reg_mse

                rank_pred = 1.0 + 4.0 * torch.sigmoid(rank_logits)
                loss_rank_point = loss_rank_point_fn(rank_pred, labels_rank)
                loss_rank_pair = pairwise_rank_loss(rank_logits, labels_rank, user_idx)
                loss_rank = (1.0 - lambda_pairwise) * loss_rank_point + lambda_pairwise * loss_rank_pair

                conf_target = _derive_conf_target(reg_logits, rank_logits, labels_reg, labels_rank)
                if epoch == 0 and batch_idx == 0:
                    _ensure_finite_tensor("train conf_target", conf_target)
                loss_conf = loss_conf_fn(conf_logits, conf_target)

                loss = loss_reg + lambda_rank * loss_rank + lambda_conf * loss_conf
                _ensure_finite_tensor("train loss", loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                compute_seconds += perf_counter() - compute_t0
                batch_loop_end = perf_counter()
                continue
            scaler.step(optimizer)
            scaler.update()
            compute_seconds += perf_counter() - compute_t0

            total_loss += float(loss.item())
            total_reg_loss += float(loss_reg.item())
            total_reg_bce += float(loss_reg_bce.item())
            total_reg_mse += float(loss_reg_mse.item())
            total_rank_point += float(loss_rank_point.item())
            total_rank_pair += float(loss_rank_pair.item())
            total_conf_loss += float(loss_conf.item())
            n_batches += 1
            samples_seen += int(labels_reg.shape[0])

            if collect_train_metrics:
                metric_rows = _append_metric_batch(
                    preds_reg_all,
                    labels_reg_all,
                    preds_rank_all,
                    labels_rank_all,
                    user_idx_all,
                    reg_prob=reg_prob,
                    labels_reg=labels_reg,
                    rank_pred=rank_pred,
                    labels_rank=labels_rank,
                    user_idx=user_idx,
                    metrics_max_samples=metric_limit,
                    collected_rows=metric_rows,
                )
            if progress_callback is not None:
                batch_number = int(batch_idx + 1)
                is_last_batch = total_train_batches is not None and batch_number >= total_train_batches
                if batch_number == 1 or batch_number % callback_interval == 0 or is_last_batch:
                    batch_fraction = (
                        batch_number / max(total_train_batches, 1)
                        if total_train_batches
                        else 0.0
                    )
                    progress_callback(
                        {
                            "event": "train_batch",
                            "epoch": int(epoch + 1),
                            "num_epochs": int(num_epochs),
                            "batch": batch_number,
                            "train_batches": int(total_train_batches or 0),
                            "percent": float(
                                100.0
                                * min(
                                    1.0,
                                    (epoch + batch_fraction) / max(num_epochs, 1),
                                )
                            ),
                            "avg_loss": float(total_loss / max(n_batches, 1)),
                            "avg_reg_loss": float(total_reg_loss / max(n_batches, 1)),
                            "avg_rank_point_loss": float(total_rank_point / max(n_batches, 1)),
                            "avg_rank_pair_loss": float(total_rank_pair / max(n_batches, 1)),
                            "avg_conf_loss": float(total_conf_loss / max(n_batches, 1)),
                            "samples_seen": int(samples_seen),
                            "samples_per_second": float(samples_seen / max(perf_counter() - train_t0, 1e-6)),
                        }
                    )
            batch_loop_end = perf_counter()

        train_reg_metrics = _compute_reg_metrics(preds_reg_all, labels_reg_all)
        train_rank_metrics = _compute_rank_metrics(preds_rank_all, labels_rank_all, user_idx_all)
        denom = max(n_batches, 1)
        train_seconds = perf_counter() - train_t0
        if verbose_stdout:
            print(
                f"Train - Loss: {total_loss / denom:.4f} | "
                f"RegLoss: {total_reg_loss / denom:.4f} | "
                f"RegBCE: {total_reg_bce / denom:.4f} | "
                f"RegMSE: {total_reg_mse / denom:.4f} | "
                f"RankPoint: {total_rank_point / denom:.4f} | "
                f"RankPair: {total_rank_pair / denom:.4f} | "
                f"ConfLoss: {total_conf_loss / denom:.4f}"
            )
            if collect_train_metrics:
                print(
                    f"  REG - MAE: {train_reg_metrics['mae_reg']:.4f}, "
                    f"RMSE: {train_reg_metrics['rmse_reg']:.4f}, "
                    f"Brier: {train_reg_metrics['brier_reg']:.4f}, "
                    f"ECE: {train_reg_metrics['ece_reg']:.4f}"
                )
                print(
                    f"  RANK - MAE@1-5: {train_rank_metrics['mae_rank']:.4f}, "
                    f"RMSE@1-5: {train_rank_metrics['rmse_rank']:.4f}, "
                    f"Spearman: {train_rank_metrics['spearman_rank']:.4f}, "
                    f"NDCG@10: {train_rank_metrics['ndcg10_rank']:.4f}"
                )
            print(
                f"  TIMING - train: {train_seconds:.1f}s, samples/s: {samples_seen / max(train_seconds, 1e-6):.1f}, "
                f"data wait: {data_wait_seconds:.1f}s, host->device: {host_to_device_seconds:.1f}s, "
                f"compute: {compute_seconds:.1f}s, metric rows: {metric_rows}"
            )

        should_validate = (
            val_loader is not None
            and ((epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs)
        )
        validation_seconds = 0.0
        val_loss = None
        val_metrics = None
        if should_validate:
            validation_t0 = perf_counter()
            val_loss, val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                lambda_rank=lambda_rank,
                lambda_pairwise=lambda_pairwise,
                lambda_conf=lambda_conf,
                metrics_max_samples=metric_limit,
                verbose_stdout=verbose_stdout,
            )
            validation_seconds = perf_counter() - validation_t0
            if verbose_stdout:
                print(
                    f"Validation - Loss: {val_loss:.4f} | "
                    f"RegLoss: {val_metrics['loss_reg']:.4f} | "
                    f"RegBCE: {val_metrics['loss_reg_bce']:.4f} | "
                    f"RegMSE: {val_metrics['loss_reg_mse']:.4f} | "
                    f"RankPoint: {val_metrics['loss_rank_point']:.4f} | "
                    f"RankPair: {val_metrics['loss_rank_pair']:.4f} | "
                    f"ConfLoss: {val_metrics['loss_conf']:.4f}"
                )
                print(
                    f"  REG - MAE: {val_metrics['mae_reg']:.4f}, "
                    f"RMSE: {val_metrics['rmse_reg']:.4f}, "
                    f"Brier: {val_metrics['brier_reg']:.4f}, "
                    f"ECE: {val_metrics['ece_reg']:.4f}"
                )
                print(
                    f"  RANK - MAE@1-5: {val_metrics['mae_rank']:.4f}, "
                    f"RMSE@1-5: {val_metrics['rmse_rank']:.4f}, "
                    f"Spearman: {val_metrics['spearman_rank']:.4f}, "
                    f"NDCG@10: {val_metrics['ndcg10_rank']:.4f}"
                )
                print(f"  TIMING - validation: {validation_seconds:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "early_stopping",
                            "epoch": int(epoch + 1),
                            "num_epochs": int(num_epochs),
                            "best_epoch": int(best_epoch) if best_epoch is not None else None,
                            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                            "epochs_without_improvement": int(epochs_no_improve),
                            "percent": float(100.0 * (epoch + 1) / max(num_epochs, 1)),
                        }
                    )
                if verbose_stdout:
                    print("Early stopping triggered.")
                break

        epoch_seconds = perf_counter() - epoch_t0
        if progress_callback is not None:
            progress_payload = {
                "event": "epoch_completed",
                "epoch": int(epoch + 1),
                "num_epochs": int(num_epochs),
                "percent": float(100.0 * (epoch + 1) / max(num_epochs, 1)),
                "train_loss": float(total_loss / max(n_batches, 1)),
                "train_reg_loss": float(total_reg_loss / max(n_batches, 1)),
                "train_rank_point_loss": float(total_rank_point / max(n_batches, 1)),
                "train_rank_pair_loss": float(total_rank_pair / max(n_batches, 1)),
                "train_conf_loss": float(total_conf_loss / max(n_batches, 1)),
                "train_reg_metrics": train_reg_metrics,
                "train_rank_metrics": train_rank_metrics,
                "train_seconds": float(train_seconds),
                "epoch_seconds": float(epoch_seconds),
                "validation_seconds": float(validation_seconds),
                "samples_seen": int(samples_seen),
                "samples_per_second": float(samples_seen / max(train_seconds, 1e-6)),
                "metric_rows": int(metric_rows),
                "metric_sample_limit": int(metric_limit) if metric_limit is not None else None,
                "validation_ran": bool(should_validate),
                "data_wait_seconds": float(data_wait_seconds),
                "host_to_device_seconds": float(host_to_device_seconds),
                "compute_seconds": float(compute_seconds),
            }
            if should_validate and val_metrics is not None and val_loss is not None:
                progress_payload.update(
                    {
                        "val_loss": float(val_loss),
                        "val_metrics": val_metrics,
                        "best_epoch": int(best_epoch) if best_epoch is not None else None,
                        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
                        "epochs_without_improvement": int(epochs_no_improve),
                    }
                )
            progress_callback(progress_payload)

        if GC_EVERY_EPOCH:
            gc.collect()
        if EMPTY_CUDA_CACHE_EVERY_EPOCH:
            torch.cuda.empty_cache()

    if val_loader is not None and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose_stdout:
            print(
                f"Restored best checkpoint from epoch {best_epoch} "
                f"(val loss: {best_val_loss:.4f})."
            )

    return {
        "best_val_loss": best_val_loss if best_state_dict is not None else None,
        "best_epoch": best_epoch if best_state_dict is not None else None,
    }


def validate(model, dataloader, device, lambda_rank=0.3, lambda_pairwise=0.5, lambda_conf=0.1, metrics_max_samples=0, verbose_stdout=True):
    model.eval()
    loss_reg_fn = nn.BCEWithLogitsLoss()
    loss_reg_mse_fn = nn.MSELoss()
    loss_rank_point_fn = nn.SmoothL1Loss()
    loss_conf_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_reg_bce = 0.0
    total_reg_mse = 0.0
    total_rank_point = 0.0
    total_rank_pair = 0.0
    total_conf_loss = 0.0
    n_batches = 0
    preds_reg_all, labels_reg_all = [], []
    preds_rank_all, labels_rank_all, user_idx_all = [], [], []
    metric_limit = int(metrics_max_samples) if metrics_max_samples and metrics_max_samples > 0 else None
    metric_rows = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False, disable=not verbose_stdout)):
            batch = _move_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                inputs = _build_inputs(
                    batch,
                    model,
                    strict_branches=True,
                    required_branches=getattr(model, "required_branches", None),
                )
                if batch_idx == 0:
                    for key, value in inputs.items():
                        _ensure_finite_nested(f"validation inputs[{key}]", value)
                reg_logits, rank_logits, conf_logits = model(inputs)

                reg_logits = reg_logits.view(-1)
                rank_logits = rank_logits.view(-1)
                conf_logits = conf_logits.view(-1)
                labels_reg = batch["label_reg"].view(-1)
                labels_rank = batch["label_rank"].view(-1).clamp(1.0, 5.0)
                user_idx = batch["user_idx"].view(-1)
                if batch_idx == 0:
                    _ensure_finite_tensor("validation labels_reg", labels_reg)
                    _ensure_finite_tensor("validation labels_rank", labels_rank)
                _ensure_finite_tensor("validation reg_logits", reg_logits)
                _ensure_finite_tensor("validation rank_logits", rank_logits)
                _ensure_finite_tensor("validation conf_logits", conf_logits)

                if labels_reg.numel() == 0:
                    continue

                reg_prob = torch.sigmoid(reg_logits)
                loss_reg_bce = loss_reg_fn(reg_logits, labels_reg)
                loss_reg_mse = loss_reg_mse_fn(reg_prob, labels_reg)
                loss_reg = 0.7 * loss_reg_bce + 0.3 * loss_reg_mse
                rank_pred = 1.0 + 4.0 * torch.sigmoid(rank_logits)
                loss_rank_point = loss_rank_point_fn(rank_pred, labels_rank)
                loss_rank_pair = pairwise_rank_loss(rank_logits, labels_rank, user_idx)
                loss_rank = (1.0 - lambda_pairwise) * loss_rank_point + lambda_pairwise * loss_rank_pair
                conf_target = _derive_conf_target(reg_logits, rank_logits, labels_reg, labels_rank)
                if batch_idx == 0:
                    _ensure_finite_tensor("validation conf_target", conf_target)
                loss_conf = loss_conf_fn(conf_logits, conf_target)
                loss = loss_reg + lambda_rank * loss_rank + lambda_conf * loss_conf
                _ensure_finite_tensor("validation loss", loss)

            total_loss += float(loss.item())
            total_reg_loss += float(loss_reg.item())
            total_reg_bce += float(loss_reg_bce.item())
            total_reg_mse += float(loss_reg_mse.item())
            total_rank_point += float(loss_rank_point.item())
            total_rank_pair += float(loss_rank_pair.item())
            total_conf_loss += float(loss_conf.item())
            n_batches += 1

            metric_rows = _append_metric_batch(
                preds_reg_all,
                labels_reg_all,
                preds_rank_all,
                labels_rank_all,
                user_idx_all,
                reg_prob=reg_prob,
                labels_reg=labels_reg,
                rank_pred=rank_pred,
                labels_rank=labels_rank,
                user_idx=user_idx,
                metrics_max_samples=metric_limit,
                collected_rows=metric_rows,
            )

    reg_metrics = _compute_reg_metrics(preds_reg_all, labels_reg_all)
    rank_metrics = _compute_rank_metrics(preds_rank_all, labels_rank_all, user_idx_all)
    denom = max(n_batches, 1)

    metrics = {
        "loss_reg": total_reg_loss / denom,
        "loss_reg_bce": total_reg_bce / denom,
        "loss_reg_mse": total_reg_mse / denom,
        "loss_rank_point": total_rank_point / denom,
        "loss_rank_pair": total_rank_pair / denom,
        "loss_conf": total_conf_loss / denom,
        "mae_reg": reg_metrics["mae_reg"],
        "rmse_reg": reg_metrics["rmse_reg"],
        "brier_reg": reg_metrics["brier_reg"],
        "ece_reg": reg_metrics["ece_reg"],
        "mae_rank": rank_metrics["mae_rank"],
        "rmse_rank": rank_metrics["rmse_rank"],
        "spearman_rank": rank_metrics["spearman_rank"],
        "ndcg10_rank": rank_metrics["ndcg10_rank"],
    }
    return total_loss / denom, metrics


def fit_reg_isotonic_calibrator(model, dataloader, max_samples=300000, verbose_stdout=True):
    _configure_torch_backends()
    if dataloader is None:
        return None

    device = torch.device("cuda")

    was_training = model.training
    model.eval()

    preds_all = []
    labels_all = []
    collected_rows = 0
    collect_limit = int(max_samples) if max_samples and max_samples > 0 else None

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Calibrating", leave=False, disable=not verbose_stdout):
            batch = _move_to_device(batch, device)
            inputs = _build_inputs(
                batch,
                model,
                strict_branches=True,
                required_branches=getattr(model, "required_branches", None),
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                reg_logits, _, _ = model(inputs)

            reg_prob = torch.sigmoid(reg_logits.view(-1)).float()
            labels_reg = batch["label_reg"].view(-1)
            if labels_reg.numel() == 0:
                continue

            preds_all.append(reg_prob.detach().cpu().numpy())
            labels_all.append(labels_reg.detach().cpu().numpy())
            collected_rows += int(labels_reg.shape[0])
            if collect_limit is not None and collected_rows >= collect_limit:
                break

    if was_training:
        model.train()

    if not preds_all or not labels_all:
        return None

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    preds, labels = clean_nan(preds, labels)
    if preds.size < 100 or np.unique(preds).size < 5:
        return None

    if preds.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(preds.size, size=max_samples, replace=False)
        preds = preds[idx]
        labels = labels[idx]

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(preds, labels)
    preds_cal = iso.predict(preds)

    calibrator = {
        "type": "isotonic",
        "x_thresholds": [float(x) for x in iso.X_thresholds_.tolist()],
        "y_thresholds": [float(y) for y in iso.y_thresholds_.tolist()],
        "n_samples": int(preds.size),
        "mae_before": float(mean_absolute_error(labels, preds)),
        "mae_after": float(mean_absolute_error(labels, preds_cal)),
        "brier_before": float(np.mean((preds - labels) ** 2)),
        "brier_after": float(np.mean((preds_cal - labels) ** 2)),
        "ece_before": float(expected_calibration_error(labels, preds)),
        "ece_after": float(expected_calibration_error(labels, preds_cal)),
    }
    return calibrator
