from typing import Any, Optional


def wb_init(
    enabled: bool,
    project: Optional[str],
    entity: Optional[str],
    tags: Optional[str],
    run_name: Optional[str],
    offline: bool,
    config: dict,
) -> Any | None:
    if not enabled:
        return None
    try:
        import os as _os
        import wandb as _wandb  # type: ignore
        if offline:
            _os.environ.setdefault("WANDB_MODE", "offline")
        tag_list = [t.strip() for t in (tags or "").split(",") if t and t.strip()]
        run = _wandb.init(
            project=project or "puffer-bandits",
            entity=entity,
            name=run_name,
            config=config,
            tags=tag_list,
            reinit=True,
        )
        try:
            print(f"[wandb] Run URL: {run.url}")
        except Exception:
            pass
        return run
    except Exception:
        return None


def wb_log(run: Any | None, data: dict, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        import wandb as _wandb  # type: ignore
        _wandb.log(data, step=step)
    except Exception:
        pass


def wb_finish(run: Any | None) -> None:
    if run is None:
        return
    try:
        import wandb as _wandb  # type: ignore
        _wandb.finish()
    except Exception:
        pass
