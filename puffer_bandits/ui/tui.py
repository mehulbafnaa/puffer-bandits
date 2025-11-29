
import math
import numpy as np
from typing import Optional, Sequence, Dict, Any
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text


def _sparkline(values: Sequence[float], width: int = 40) -> str:
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    v = list(values[-width:])
    vmin = min(v)
    vmax = max(v)
    if math.isclose(vmax, vmin):
        return chars[0] * len(v)
    out = []
    for x in v:
        idx = int((x - vmin) / (vmax - vmin) * (len(chars) - 1) + 1e-9)
        out.append(chars[idx])
    return "".join(out)


class RichTUI:
    """Rich-based terminal UI for live bandit runs.

    Shows KPIs, progress, sparkline of mean reward, and arm selection histogram.
    """

    def __init__(self, k: int, T: int, device_desc: str, title: str = "", theme: str = "dark") -> None:

        self.k = int(k)
        self.T = int(T)
        self._means: list[float] = []
        self._pcts: list[float] = []
        self._regrets: list[float] = []
        self._cum_counts = [0] * k
        self._cum_regret_by_arm: list[float] = [0.0] * k
        self._theme = theme
        self._last_mean: Optional[float] = None

        self.Layout = Layout  # type: ignore[attr-defined]
        self.Panel = Panel  # type: ignore[attr-defined]
        self.Table = Table  # type: ignore[attr-defined]
        self.Progress = Progress  # type: ignore[attr-defined]
        self.BarColumn = BarColumn  # type: ignore[attr-defined]
        self.TextColumn = TextColumn  # type: ignore[attr-defined]
        self.TimeRemainingColumn = TimeRemainingColumn  # type: ignore[attr-defined]
        self.Text = Text  # type: ignore[attr-defined]

        self.layout = layout = self.Layout()
        layout.split(
            self.Layout(name="header", size=3),
            self.Layout(name="body"),
            self.Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            self.Layout(name="left"),
            self.Layout(name="right"),
        )
        layout["body"]["left"].split(
            self.Layout(name="kpi"),
            self.Layout(name="trends"),
        )
        layout["body"]["right"].split(
            self.Layout(name="hist_step"),
            self.Layout(name="hist_cum"),
            self.Layout(name="values"),
            self.Layout(name="perf"),
            self.Layout(name="regret_top"),
        )

        self._progress = self.Progress(
            self.TextColumn("[bold cyan]Step"),
            self.BarColumn(bar_width=None),
            self.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            self.TimeRemainingColumn(),
            expand=True,
        )
        self._task = self._progress.add_task("run", total=T)

        header = "puffer-bandits • Native Puffer"
        if title:
            header = f"{header} • {title}"
        layout["header"].update(self.Panel(f"[bold magenta]{header} • Device: {device_desc}"))
        layout["footer"].update(self._progress)

        self.live = Live(layout, auto_refresh=False, refresh_per_second=20)
        self.live.start()

    def _kpi_panel(self, t: int, mean_r: float, pct_opt: float, regret: float, mem: Dict[str, Any], speed_sps: Optional[float]) -> "Panel":
        tbl = self.Table(show_header=False, box=None, pad_edge=False)
        tbl.add_column("metric", style="bold green")
        tbl.add_column("value")
        arrow = ""
        if self._last_mean is not None:
            arrow = "▲" if mean_r >= self._last_mean else "▼"
        self._last_mean = mean_r
        self._means.append(mean_r)
        self._pcts.append(pct_opt)
        self._regrets.append(regret)
        spark = _sparkline(self._means, width=40)
        mem_str = ", ".join(f"{k}={v/1e6:.1f}MB" for k, v in mem.items()) if mem else ""
        tbl.add_row("t/T", f"{t}/{self.T}")
        tbl.add_row("mean reward", f"{mean_r:.4f} {arrow}  {spark}")
        tbl.add_row("% optimal", f"{pct_opt:.2f}%")
        tbl.add_row("regret", f"{regret:.4f}")
        if speed_sps is not None:
            tbl.add_row("speed", f"{speed_sps:.1f} steps/s")
        if mem_str:
            tbl.add_row("mem", mem_str)
        return self.Panel(tbl, title="KPIs", border_style="cyan")

    def _hist_panel(self, actions: Optional[Sequence[int]]) -> "Panel":
        tbl = self.Table(show_header=True, header_style="bold yellow")
        tbl.add_column("arm")
        tbl.add_column("count", justify="right")
        tbl.add_column("pct", justify="right")
        if actions:
            a = np.asarray(actions)
            k = self.k
            counts = np.bincount(a, minlength=k)
            total = counts.sum() if counts.sum() > 0 else 1
            # Show top 6 arms
            top_idx = np.argsort(-counts)[:6]
            for i in top_idx:
                pct = 100.0 * float(counts[i]) / float(total)
                tbl.add_row(str(int(i)), str(int(counts[i])), f"{pct:5.1f}%")
        return self.Panel(tbl, title="Selected Arms (this step)", border_style="yellow")

    def _hist_cum_panel(self) -> "Panel":
        tbl = self.Table(show_header=True, header_style="bold yellow")
        tbl.add_column("arm")
        tbl.add_column("total", justify="right")
        tbl.add_column("pct", justify="right")

        counts = np.asarray(self._cum_counts, dtype=int)
        total = counts.sum() if counts.sum() > 0 else 1
        if total > 0:
            top_idx = np.argsort(-counts)[:6]
            for i in top_idx:
                pct = 100.0 * float(counts[i]) / float(total)
                tbl.add_row(str(int(i)), str(int(counts[i])), f"{pct:5.1f}%")
        return self.Panel(tbl, title="Cumulative Arm Selection", border_style="yellow")

    def _values_panel(self, values: Optional[Sequence[tuple]], labels: Optional[Dict[str, str]] = None) -> "Panel":
        tbl = self.Table(show_header=True, header_style="bold cyan")
        est_label = (labels or {}).get("est", "est p")
        true_label = (labels or {}).get("true", "true p")
        extra_label = (labels or {}).get("extra", "conf")
        tbl.add_column("arm")
        tbl.add_column(est_label, justify="right")
        tbl.add_column(true_label, justify="right")
        has_conf = False
        # detect conf column
        if values and len(values) > 0 and len(values[0]) == 4:
            tbl.add_column(extra_label, justify="right")
            has_conf = True
        if values:
            # show up to 6 rows
            for row in list(values)[:6]:
                if has_conf:
                    i, est, truep, conf = row  # type: ignore[misc]
                else:
                    i, est, truep = row  # type: ignore[misc]
                est_s = f"{est:0.3f}"
                true_s = "" if truep is None else f"{truep:0.3f}"
                if has_conf:
                    conf_s = f"{conf:0.3f}" if conf is not None else ""
                    tbl.add_row(str(int(i)), est_s, true_s, conf_s)
                else:
                    tbl.add_row(str(int(i)), est_s, true_s)
        return self.Panel(tbl, title="Arm Value Estimates (env 0)", border_style="cyan")

    def _perf_panel(self, last_ms: Optional[float], ewma_ms: Optional[float], sps: Optional[float]) -> "Panel":
        tbl = self.Table(show_header=False, box=None)
        tbl.add_column("metric", style="bold")
        tbl.add_column("value")
        if last_ms is not None:
            tbl.add_row("last ms/step", f"{last_ms:.2f}")
        if ewma_ms is not None:
            tbl.add_row("avg ms/step", f"{ewma_ms:.2f}")
        if sps is not None:
            tbl.add_row("steps/s", f"{sps:.1f}")
        return self.Panel(tbl, title="Performance", border_style="magenta")

    def _regret_top_panel(self) -> "Panel":
        tbl = self.Table(show_header=True, header_style="bold red")
        tbl.add_column("arm")
        tbl.add_column("cumulative regret", justify="right")
        tbl.add_column("pct", justify="right")

        r = np.asarray(self._cum_regret_by_arm, dtype=float)
        total = float(r.sum()) if r.sum() > 0 else 1.0
        top_idx = np.argsort(-r)[:6]
        for i in top_idx:
            pct = 100.0 * (float(r[i]) / total)
            tbl.add_row(str(int(i)), f"{float(r[i]):.3f}", f"{pct:5.1f}%")
        return self.Panel(tbl, title="Top Regret Contributors", border_style="red")

    def update(self, *, t: int, mean_r: float, pct_opt: float, regret: float,
               actions: Optional[Sequence[int]], mem: Dict[str, Any],
               values: Optional[Sequence[tuple]] = None,
               speed_sps: Optional[float] = None,
               cum_counts: Optional[Sequence[int]] = None,
               cum_regret_by_arm: Optional[Sequence[float]] = None,
               last_ms: Optional[float] = None,
               ewma_ms: Optional[float] = None,
               values_labels: Optional[Dict[str, str]] = None) -> None:
        # Update progress
        self._progress.update(self._task, completed=t)
        # Update cumulative counts if provided by runner
        if cum_counts is not None:
            try:
                self._cum_counts = list(cum_counts)
            except Exception:
                pass
        if cum_regret_by_arm is not None:
            try:
                self._cum_regret_by_arm = list(cum_regret_by_arm)
            except Exception:
                pass
        # Panels
        self.layout["body"]["left"]["kpi"].update(self._kpi_panel(t, mean_r, pct_opt, regret, mem, speed_sps))
        # Trends
        trend_tbl = self.Table(show_header=False, box=None)
        trend_tbl.add_column("metric", style="bold")
        trend_tbl.add_column("spark")
        trend_tbl.add_row("mean", _sparkline(self._means, width=50))
        trend_tbl.add_row("%opt", _sparkline(self._pcts, width=50))
        trend_tbl.add_row("regret", _sparkline(self._regrets, width=50))
        self.layout["body"]["left"]["trends"].update(self.Panel(trend_tbl, title="Trends", border_style="green"))

        self.layout["body"]["right"]["hist_step"].update(self._hist_panel(actions))
        self.layout["body"]["right"]["hist_cum"].update(self._hist_cum_panel())
        self.layout["body"]["right"]["values"].update(self._values_panel(values, labels=values_labels))
        self.layout["body"]["right"]["perf"].update(self._perf_panel(last_ms, ewma_ms, sps=speed_sps))
        self.layout["body"]["right"]["regret_top"].update(self._regret_top_panel())
        self.live.update(self.layout, refresh=True)

    def stop(self) -> None:
        try:
            self.live.stop()
        except Exception:
            pass
