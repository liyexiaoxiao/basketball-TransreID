import torch


def make_optimizer(cfg, model, center_criterion):
    """
    Build optimizer and center optimizer.

    For AdamW:
      - Each param group carries its own lr and weight_decay.
      - The optimizer-level lr/weight_decay serve as defaults.
      - Betas are taken from SOLVER.ADAM_BETAS (default (0.9, 0.999)).

    For SGD:
      - momentum is applied globally; per-group lr/weight_decay still work.

    Center loss always uses SGD (per original paper).
    """
    params = []
    use_adamw = (cfg.SOLVER.OPTIMIZER_NAME == 'AdamW')

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # ── Bias / Norm: optionally different lr and no weight decay ────────
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        # ── Classifier heads: optionally higher lr (SGD trick, less needed for AdamW) ──
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # ── Build main optimizer ─────────────────────────────────────────────────
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
            params,
            momentum=cfg.SOLVER.MOMENTUM,
        )
        print(f"Optimizer: SGD (lr={cfg.SOLVER.BASE_LR}, momentum={cfg.SOLVER.MOMENTUM})")

    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        betas = tuple(cfg.SOLVER.ADAM_BETAS)
        # Per-group lr/weight_decay override the defaults passed here
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=betas,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        print(f"Optimizer: AdamW (lr={cfg.SOLVER.BASE_LR}, betas={betas}, wd={cfg.SOLVER.WEIGHT_DECAY})")

    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        print(f"Optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")

    # ── Center optimizer (always SGD) ────────────────────────────────────────
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
