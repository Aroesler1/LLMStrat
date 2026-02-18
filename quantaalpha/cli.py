"""
LLMStrat CLI entry.

Commands:
  quantaalpha mine       - run factor mining
  quantaalpha backtest   - run backtest
  quantaalpha paper      - run paper trading runtime
  quantaalpha trade      - trading subcommands
  quantaalpha ui         - start log Web UI
  quantaalpha health_check - environment health check
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env (prefer project root, fallback to cwd)
_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(".env")


def _ensure_conda_default_env() -> None:
    """rdagent requires CONDA_DEFAULT_ENV even when running from a venv."""
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return
    fallback = os.environ.get("CONDA_ENV_NAME")
    if not fallback:
        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            fallback = Path(venv_path).name
    if not fallback:
        fallback = "quantaalpha"
    os.environ["CONDA_DEFAULT_ENV"] = fallback


_ensure_conda_default_env()


def _ensure_timeout_shim_on_path() -> None:
    """Add portable timeout shim to PATH for macOS/local runs."""
    shim_dir = _project_root / "scripts" / "bin"
    if not shim_dir.exists():
        return
    current_path = os.environ.get("PATH", "")
    shim_dir_str = str(shim_dir)
    if any(p == shim_dir_str for p in current_path.split(":")):
        pass
    else:
        os.environ["PATH"] = f"{shim_dir_str}:{current_path}" if current_path else shim_dir_str

    # rdagent LocalEnv builds PATH from conda bin paths; ensure timeout exists there too.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    timeout_shim = shim_dir / "timeout"
    if conda_prefix and timeout_shim.exists():
        conda_timeout = Path(conda_prefix) / "bin" / "timeout"
        if not conda_timeout.exists():
            try:
                conda_timeout.symlink_to(timeout_shim)
            except OSError:
                # Non-fatal: PATH shim above still works for normal local execution.
                pass


_ensure_timeout_shim_on_path()

import fire
from quantaalpha.pipeline.factor_mining import main as mine
from quantaalpha.pipeline.factor_backtest import main as backtest
from quantaalpha.app.utils.health_check import health_check
from quantaalpha.app.utils.info import collect_info


class TradeCLI:
    """Trading runtime commands."""

    @staticmethod
    def _engine(config_path: str = "configs/trading.yaml", paper: bool | None = None, overrides: dict | None = None):
        from quantaalpha.trading.engine import TradingEngine

        return TradingEngine.from_yaml(
            config_path=config_path,
            paper=paper,
            overrides=overrides,
        )

    def start(
        self,
        config_path: str = "configs/trading.yaml",
        paper: bool = False,
        once: bool = False,
        no_scheduler: bool = False,
        dry_run: bool | None = None,
    ):
        overrides = {}
        if dry_run is not None:
            overrides["execution"] = {"dry_run": bool(dry_run)}
        engine = self._engine(config_path=config_path, paper=paper, overrides=overrides or None)
        return engine.start(run_once=once, with_scheduler=(not no_scheduler))

    def paper(
        self,
        config_path: str = "configs/paper.yaml",
        once: bool = False,
        no_scheduler: bool = False,
        dry_run: bool | None = None,
    ):
        return self.start(
            config_path=config_path,
            paper=True,
            once=once,
            no_scheduler=no_scheduler,
            dry_run=dry_run,
        )

    def status(
        self,
        config_path: str = "configs/trading.yaml",
        paper: bool = True,
    ):
        engine = self._engine(config_path=config_path, paper=paper)
        return engine.status()

    def rebalance(
        self,
        config_path: str = "configs/trading.yaml",
        paper: bool = True,
        dry_run: bool | None = None,
    ):
        overrides = {}
        if dry_run is not None:
            overrides["execution"] = {"dry_run": bool(dry_run)}
        engine = self._engine(config_path=config_path, paper=paper, overrides=overrides or None)
        return engine.rebalance_once(reason="manual").to_dict()

    def stop(
        self,
        config_path: str = "configs/trading.yaml",
        paper: bool = True,
        flatten: bool = False,
    ):
        engine = self._engine(config_path=config_path, paper=paper)
        return engine.stop(flatten=flatten)


def paper(config_path: str = "configs/paper.yaml", once: bool = False, no_scheduler: bool = False, dry_run: bool | None = None):
    return TradeCLI().paper(config_path=config_path, once=once, no_scheduler=no_scheduler, dry_run=dry_run)


def trade_status(config_path: str = "configs/trading.yaml", paper: bool = True):
    return TradeCLI().status(config_path=config_path, paper=paper)


def trade_stop(config_path: str = "configs/trading.yaml", paper: bool = True, flatten: bool = False):
    return TradeCLI().stop(config_path=config_path, paper=paper, flatten=flatten)


def trade_rebalance(config_path: str = "configs/trading.yaml", paper: bool = True, dry_run: bool | None = None):
    return TradeCLI().rebalance(config_path=config_path, paper=paper, dry_run=dry_run)


def app():
    fire.Fire(
        {
            "mine": mine,
            "backtest": backtest,
            "trade": TradeCLI(),
            "paper": paper,
            "status": trade_status,
            "rebalance": trade_rebalance,
            "stop": trade_stop,
            "health_check": health_check,
            "collect_info": collect_info,
        }
    )


if __name__ == "__main__":
    app()
