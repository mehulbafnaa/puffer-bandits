
import argparse
from dataclasses import asdict
from typing import Literal

from omegaconf import OmegaConf  # type: ignore


def _load_conf(defaults: dict, args_namespace) -> dict:
    conf = OmegaConf.create(defaults)
    if args_namespace.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(args_namespace.config))
    if args_namespace.set:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args_namespace.set))
    return OmegaConf.to_container(conf, resolve=True)  # type: ignore


def _filter_keys(d: dict, keys: set[str]) -> dict:
    return {k: v for k, v in d.items() if k in keys}


def main() -> None:
    p = argparse.ArgumentParser("puffer-bandits minimal runner (config-first)")
    p.add_argument("--runner", type=str, choices=["native", "advanced", "classic"], default="native")
    p.add_argument("--config", type=str, default=None, help="YAML/TOML config file")
    p.add_argument("--set", action="append", default=None, help="Override config via dotlist, e.g., runs=1024")
    p.add_argument("--preset", type=str, default=None, help="native-only: {smoke,experiment,benchmark,neural}")
    p.add_argument("--tui", action="store_true", help="Enable Rich TUI (CLI-only; ignored in config)")
    p.add_argument("--print-config", action="store_true", help="print the resolved config for the selected runner")
    args = p.parse_args()

    if args.runner == "native":
        from .runner_puffer_native import Config as NConfig, run_with_config as nrun, PRESETS
        base = asdict(NConfig())
        # Apply preset first if provided
        conf = OmegaConf.create(base)
        if args.preset:
            if args.preset not in PRESETS:
                raise SystemExit(f"Unknown preset: {args.preset}")
            conf = OmegaConf.merge(conf, OmegaConf.create(PRESETS[args.preset]))
        # Then file + dotlist
        if args.config:
            conf = OmegaConf.merge(conf, OmegaConf.load(args.config))
        if args.set:
            conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.set))
        # Enforce CLI-only for TUI
        conf["tui"] = bool(getattr(args, "tui", False))
        conf_dict = OmegaConf.to_container(conf, resolve=True)  # type: ignore
        # Filter out non-dataclass keys (like 'runner')
        nkeys = set(NConfig.__dataclass_fields__.keys())
        conf_dict = _filter_keys(conf_dict, nkeys)  # type: ignore[arg-type]
        cfg = NConfig(**conf_dict)  # type: ignore
        if args.print_config:
            print(OmegaConf.to_yaml(OmegaConf.create(conf_dict)))
        nrun(cfg)
        return

    if args.runner == "advanced":
        from .runner_puffer_advanced import Config as AConfig, run_with_config as arun
        base = asdict(AConfig())
        conf_dict = _load_conf(base, args)
        conf_dict["tui"] = bool(getattr(args, "tui", False))
        akeys = set(AConfig.__dataclass_fields__.keys())
        conf_dict = _filter_keys(conf_dict, akeys)  # type: ignore[arg-type]
        cfg = AConfig(**conf_dict)  # type: ignore
        if args.print_config:
            print(OmegaConf.to_yaml(OmegaConf.create(conf_dict)))
        arun(cfg)
        return

    # classic
    from .runner_puffer import Config as CConfig, run_with_config as crun
    base = asdict(CConfig())
    conf_dict = _load_conf(base, args)
    conf_dict["tui"] = bool(getattr(args, "tui", False))
    ckeys = set(CConfig.__dataclass_fields__.keys())
    conf_dict = _filter_keys(conf_dict, ckeys)  # type: ignore[arg-type]
    cfg = CConfig(**conf_dict)  # type: ignore
    if args.print_config:
        print(OmegaConf.to_yaml(OmegaConf.create(conf_dict)))
    crun(cfg)
