#!/usr/bin/env python3
"""
Backtest runner using Qlib: load factors (official/custom), compute custom factor values, train, backtest, evaluate.
Modes: official (Qlib DataLoader) or custom (expr_parser + function_lib).
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Backtest executor."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._qlib_initialized = False

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config: {self.config_path}")
        return config

    def _init_qlib(self):
        if self._qlib_initialized:
            return
        import os
        # Some restricted environments block semaphore limit syscalls used by
        # joblib's multiprocessing backend. Default to single-process mode
        # unless explicitly overridden.
        if os.environ.get("QA_ENABLE_JOBLIB_MP", "0") != "1":
            os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
        import qlib
        provider_uri = (
            os.environ.get('QLIB_DATA_DIR')
            or os.environ.get('QLIB_PROVIDER_URI')
            or self.config['data']['provider_uri']
        )
        provider_uri = os.path.expanduser(provider_uri)
        region = self.config['data'].get('region', 'cn')
        # In restricted environments, joblib multiprocessing can fail when querying
        # system semaphore limits. Use safe defaults unless explicitly overridden.
        qlib_cfg = self.config.get("qlib", {}) or {}
        kernels = int(os.environ.get("QA_QLIB_KERNELS", qlib_cfg.get("kernels", 1)))
        joblib_backend = os.environ.get(
            "QA_QLIB_JOBLIB_BACKEND", qlib_cfg.get("joblib_backend", "threading")
        )
        qlib.init(
            provider_uri=provider_uri,
            region=region,
            kernels=kernels,
            joblib_backend=joblib_backend,
        )
        self._qlib_initialized = True
        logger.info(
            f"Qlib initialized: {provider_uri} (region={region}, kernels={kernels}, joblib_backend={joblib_backend})"
        )

    def run(self,
            factor_source: Optional[str] = None,
            factor_json: Optional[List[str]] = None,
            experiment_name: Optional[str] = None,
            output_name: Optional[str] = None,
            skip_uncached: bool = False) -> Dict:
        """Run full backtest; returns metrics dict."""
        start_time_total = time.time()
        self._init_qlib()
        if factor_source:
            self.config['factor_source']['type'] = factor_source
        if factor_json:
            self.config['factor_source']['custom']['json_files'] = factor_json
        
        if output_name is None and factor_json:
            output_name = Path(factor_json[0]).stem

        exp_name = experiment_name or output_name or self.config['experiment']['name']
        rec_name = self.config['experiment']['recorder']

        print(f"\n{'='*50}")
        src = factor_json[0] if factor_json else exp_name
        print(f"Starting backtest: {src}")
        print(f"{'='*50}")

        factor_expressions, custom_factors = self._load_factors()
        print(f"[1/4] Loaded factors: Qlib {len(factor_expressions)}, custom {len(custom_factors)}")

        computed_factors = None
        if custom_factors:
            computed_factors = self._compute_custom_factors(custom_factors, skip_compute=skip_uncached)
            n_computed = len(computed_factors.columns) if computed_factors is not None and not computed_factors.empty else 0
            print(f"[2/4] Computed custom factors: {n_computed}")
        else:
            logger.debug("[2/4] No custom factors, skip")

        walk_forward_cfg = (self.config.get("walk_forward") or {})
        use_walk_forward = bool(walk_forward_cfg.get("enabled", False))

        if use_walk_forward:
            print("[3/4] Walk-forward mode enabled")
            metrics = self._run_walk_forward_backtest(
                factor_expressions=factor_expressions,
                computed_factors=computed_factors,
                exp_name=exp_name,
                output_name=output_name,
            )
        else:
            dataset = self._create_dataset(factor_expressions, computed_factors)
            print("[3/4] Dataset created")
            metrics = self._train_and_backtest(dataset, exp_name, rec_name, output_name=output_name)
        total_time = time.time() - start_time_total
        self._print_results(metrics, total_time)
        self._save_results(metrics, exp_name, factor_source or self.config['factor_source']['type'], 
                          len(factor_expressions) + len(custom_factors), total_time,
                          output_name=output_name)
        
        return metrics
    
    def _load_factors(self) -> Tuple[Dict[str, str], List[Dict]]:
        from .factor_loader import FactorLoader
        
        loader = FactorLoader(self.config)
        return loader.load_factors()
    
    def _compute_custom_factors(self, factors: List[Dict], skip_compute: bool = False) -> Optional[pd.DataFrame]:
        """Compute custom factors (expr_parser + function_lib); supports cache; loads stock data only when needed."""
        from .custom_factor_calculator import CustomFactorCalculator
        from pathlib import Path

        llm_config = self.config.get('llm', {})
        cache_dir = llm_config.get('cache_dir')
        if cache_dir:
            cache_dir = Path(cache_dir)
        auto_extract = llm_config.get('auto_extract_cache', True)
        calculator = CustomFactorCalculator(
            data_df=None,
            cache_dir=cache_dir,
            auto_extract_cache=auto_extract,
            config=self.config,
        )
        result_df = calculator.calculate_factors_batch(factors, use_cache=True, skip_compute=skip_compute)
        if result_df is None:
            logger.error("Factor computation returned None")
            return None
        if not isinstance(result_df, pd.DataFrame):
            logger.error(f"Factor computation returned wrong type: {type(result_df)}")
            return None
        
        if result_df.empty:
            logger.error("Factor computation returned empty DataFrame")
            return None
        
        if not isinstance(result_df.index, pd.MultiIndex):
            logger.warning("Factor data index is not MultiIndex, attempting fix...")
        logger.debug(f"  Factor computation done: {len(result_df.columns)} factors, {len(result_df)} rows")
        
        return result_df
    
    def _create_dataset(self, 
                       factor_expressions: Dict[str, str],
                       computed_factors: Optional[pd.DataFrame] = None):
        """Create Qlib dataset (QlibDataLoader or precomputed factors + StaticDataLoader)."""
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        has_computed_factors = False
        if computed_factors is not None:
            if isinstance(computed_factors, pd.DataFrame):
                if len(computed_factors) > 0 and len(computed_factors.columns) > 0:
                    has_computed_factors = True
                    logger.debug(f"  Precomputed factors: {len(computed_factors.columns)} factors, {len(computed_factors)} rows")
                else:
                    logger.warning(f"  Precomputed factor DataFrame is empty: {computed_factors.shape}")
            else:
                logger.warning(f"  Precomputed factor type invalid: {type(computed_factors)}")
        
        # Prefer custom factor mode when computed factors exist
        if has_computed_factors:
            logger.debug("  Using custom factor mode (precomputed)")
            return self._create_dataset_with_computed_factors(
                factor_expressions, computed_factors
            )
        
        # Qlib-only factor mode
        expressions = list(factor_expressions.values())
        names = list(factor_expressions.keys())
        
        if not expressions:
            raise ValueError("No factor expressions available. If using custom factors, ensure factor computation succeeded.")
        
        handler_config = {
            'start_time': data_config['start_time'],
            'end_time': data_config['end_time'],
            'instruments': data_config['market'],
            'data_loader': {
                'class': 'QlibDataLoader',
                'module_path': 'qlib.contrib.data.loader',
                'kwargs': {
                    'config': {
                        'feature': (expressions, names),
                        'label': ([dataset_config['label']], ['LABEL0'])
                    }
                }
            },
            'learn_processors': dataset_config['learn_processors'],
            'infer_processors': dataset_config['infer_processors']
        }
        
        dataset = DatasetH(
            handler=DataHandlerLP(**handler_config),
            segments=dataset_config['segments']
        )
        
        logger.debug(f"  Qlib mode: {len(expressions)} factors, train={dataset_config['segments']['train']}")
        
        return dataset
    
    def _create_dataset_with_computed_factors(self,
                                              factor_expressions: Dict[str, str],
                                              computed_factors: pd.DataFrame):
        """Create dataset from precomputed factors: compute label, merge with factors, use custom DataHandler."""
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandler
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        logger.debug(f"  Computed factor count: {len(computed_factors.columns)}")
        label_expr = dataset_config['label']
        label_df = self._compute_label(label_expr)

        # Strict novelty controls for computed factors:
        # correlation filtering, orthogonalization, and IC-decay pruning.
        try:
            from .factor_postprocess import FactorPostProcessor, FactorPostprocessConfig

            post_cfg = FactorPostprocessConfig.from_dict(self.config.get('factor_postprocess'))
            if post_cfg.enabled:
                # Fit selection/orthogonalization only on train+valid to avoid look-ahead.
                computed_factors = self._normalize_multiindex(computed_factors, "computed_factors_pre")
                label_df = self._normalize_multiindex(label_df, "label_pre")
                fit_index = self._build_model_selection_fit_index(computed_factors.index)
                processor = FactorPostProcessor(post_cfg)
                computed_factors = processor.process(
                    computed_factors,
                    label_df=label_df,
                    fit_index=fit_index,
                )
                logger.info(f"  Postprocessed computed factors: {len(computed_factors.columns)} columns")
                if computed_factors.empty:
                    raise ValueError("All computed factors were filtered out by factor_postprocess settings")
        except Exception as e:
            logger.warning(f"  Factor postprocess skipped due to error: {e}")
        
        all_feature_dfs = [computed_factors]
        if factor_expressions:
            logger.debug(f"  Loading {len(factor_expressions)} Qlib-compatible factors")
            qlib_factors = self._load_qlib_factors(factor_expressions)
            if qlib_factors is not None and not qlib_factors.empty:
                all_feature_dfs.append(qlib_factors)
        
        features_df = pd.concat(all_feature_dfs, axis=1)
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]
        logger.debug(f"  Total factor count: {len(features_df.columns)}")

        def _normalize_multiindex(df, df_name):
            """Ensure MultiIndex has standard (datetime, instrument) level names."""
            if not isinstance(df.index, pd.MultiIndex):
                logger.warning(f"  {df_name} index is not MultiIndex: {type(df.index)}")
                return df
            
            names = list(df.index.names)
            logger.debug(f"  {df_name} index levels: {names}, "
                        f"dtypes: {[str(df.index.get_level_values(i).dtype) for i in range(len(names))]}, "
                        f"len: {len(df)}")
            
            new_names = list(names)
            for i, name in enumerate(names):
                level_vals = df.index.get_level_values(i)
                if name == 'datetime' or name == 'date':
                    new_names[i] = 'datetime'
                elif name == 'instrument' or name == 'stock':
                    new_names[i] = 'instrument'
                elif name is None:
                    # Infer from dtype
                    if pd.api.types.is_datetime64_any_dtype(level_vals):
                        new_names[i] = 'datetime'
                    elif level_vals.dtype == object or pd.api.types.is_string_dtype(level_vals):
                        new_names[i] = 'instrument'
            
            if new_names != names:
                logger.debug(f"  {df_name} index renamed: {names} -> {new_names}")
                df.index = df.index.set_names(new_names)
            actual_names = list(df.index.names)
            if len(actual_names) == 2 and actual_names == ['instrument', 'datetime']:
                df = df.swaplevel()
                df = df.sort_index()
                logger.debug(f"  {df_name} index swapped to (datetime, instrument)")
            
            return df
        
        features_df = _normalize_multiindex(features_df, "features")
        label_df = _normalize_multiindex(label_df, "label")
        
        common_index = features_df.index.intersection(label_df.index)
        if len(common_index) == 0 and len(features_df) > 0 and len(label_df) > 0:
            logger.warning("  Index intersection empty, aligning datetime types...")
            feat_dt = features_df.index.get_level_values('datetime')
            label_dt = label_df.index.get_level_values('datetime')
            logger.debug(f"  features datetime dtype={feat_dt.dtype}, sample={feat_dt[:3].tolist()}")
            logger.debug(f"  label    datetime dtype={label_dt.dtype}, sample={label_dt[:3].tolist()}")
            
            feat_inst = features_df.index.get_level_values('instrument')
            label_inst = label_df.index.get_level_values('instrument')
            logger.debug(f"  features instrument sample={feat_inst[:3].tolist()}")
            logger.debug(f"  label    instrument sample={label_inst[:3].tolist()}")
            
            try:
                if not pd.api.types.is_datetime64_any_dtype(feat_dt):
                    features_df.index = features_df.index.set_levels(
                        pd.to_datetime(feat_dt.unique()), level='datetime'
                    )
                    logger.debug("  features datetime converted to Timestamp")
                if not pd.api.types.is_datetime64_any_dtype(label_dt):
                    label_df.index = label_df.index.set_levels(
                        pd.to_datetime(label_dt.unique()), level='datetime'
                    )
                    logger.debug("  label datetime converted to Timestamp")
            except Exception as e:
                logger.warning(f"  datetime type conversion failed: {e}")
            common_index = features_df.index.intersection(label_df.index)
            logger.debug(f"  Intersection size after align: {len(common_index)}")

        if len(common_index) == 0:
            logger.warning("  Index intersection still empty, trying merge...")
            feat_reset = features_df.reset_index()
            label_reset = label_df.reset_index()
            dt_col = 'datetime' if 'datetime' in feat_reset.columns else feat_reset.columns[0]
            inst_col = 'instrument' if 'instrument' in feat_reset.columns else feat_reset.columns[1]
            
            merged = pd.merge(
                feat_reset, label_reset,
                on=[dt_col, inst_col],
                how='inner'
            )
            logger.debug(f"  Merged rows: {len(merged)}")
            if len(merged) == 0:
                raise ValueError(
                    f"Factor and label data could not be aligned. "
                    f"features: {len(features_df)} rows, index names={list(features_df.index.names)}; "
                    f"label: {len(label_df)} rows, index names={list(label_df.index.names)}"
                )
            
            merged = merged.set_index([dt_col, inst_col])
            merged.index.names = ['datetime', 'instrument']
            
            feature_cols = [c for c in features_df.columns if c in merged.columns]
            label_cols = [c for c in label_df.columns if c in merged.columns]
            features_df = merged[feature_cols]
            label_df = merged[label_cols]
        else:
            features_df = features_df.loc[common_index]
            label_df = label_df.loc[common_index]
        
        logger.debug(f"  Data rows: {len(features_df)}")
        if len(features_df) == 0:
            raise ValueError("No rows after index alignment; cannot run backtest")
        combined_df = pd.concat([features_df, label_df], axis=1)
        from qlib.data.dataset.processor import Fillna, ProcessInf, CSRankNorm, DropnaLabel
        feature_cols = list(features_df.columns)
        label_cols = list(label_df.columns)
        combined_df[feature_cols] = combined_df[feature_cols].fillna(0)
        combined_df[feature_cols] = combined_df[feature_cols].replace([np.inf, -np.inf], 0)
        dt_level = combined_df.index.names[0] if combined_df.index.names[0] else 0
        for col in feature_cols:
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        combined_df = combined_df.dropna(subset=label_cols)
        for col in label_cols:
            combined_df[col] = combined_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0
            )
        
        logger.debug(f"  Rows after preprocessing: {len(combined_df)}")
        feature_tuples = [('feature', col) for col in feature_cols]
        label_tuples = [('label', col) for col in label_cols]
        
        combined_df_multi = combined_df.copy()
        combined_df_multi.columns = pd.MultiIndex.from_tuples(
            feature_tuples + label_tuples
        )
        
        class PrecomputedDataHandler(DataHandler):
            """DataHandler for precomputed data."""
            
            def __init__(self, data_df, segments):
                self._data = data_df
                self._segments = segments
            
            @property
            def data_loader(self):
                return None
            
            @property
            def instruments(self):
                try:
                    return list(self._data.index.get_level_values('instrument').unique())
                except KeyError:
                    return list(self._data.index.get_level_values(1).unique())
            
            def fetch(self, selector=None, level='datetime', col_set='feature',
                     data_key=None, squeeze=False, proc_func=None):
                if col_set in ('feature', 'label'):
                    result = self._data[col_set].copy()
                elif col_set == '__all' or col_set is None:
                    result = self._data.copy()
                else:
                    if isinstance(col_set, (list, tuple)):
                        result = self._data[list(col_set)].copy()
                    else:
                        result = self._data.copy()
                if selector is not None:
                    try:
                        dates = result.index.get_level_values('datetime')
                    except KeyError:
                        dates = result.index.get_level_values(0)
                    if isinstance(selector, tuple) and len(selector) == 2:
                        start, end = selector
                        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                        result = result.loc[mask]
                    elif isinstance(selector, slice):
                        start = selector.start
                        end = selector.stop
                        if start is not None and end is not None:
                            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                            result = result.loc[mask]
                
                if squeeze and result.shape[1] == 1:
                    result = result.iloc[:, 0]
                
                return result
            
            def get_cols(self, col_set='feature'):
                if col_set in self._data.columns.get_level_values(0):
                    return list(self._data[col_set].columns)
                return list(self._data.columns.get_level_values(1))
            
            def setup_data(self, **kwargs):
                pass
            
            def config(self, **kwargs):
                pass
        
        handler = PrecomputedDataHandler(combined_df_multi, dataset_config['segments'])
        dataset = DatasetH(
            handler=handler,
            segments=dataset_config['segments']
        )
        
        logger.debug(f"  Custom factor mode: {len(feature_cols)} factors, {len(combined_df)} rows, train={dataset_config['segments']['train']}")
        
        return dataset
    
    def _compute_label(self, label_expr: str) -> pd.DataFrame:
        """Compute label using Qlib (label requires look-ahead)."""
        from qlib.data import D
        
        data_config = self.config['data']
        
        logger.debug(f"  Label expr: {label_expr}")
        
        stock_list = D.instruments(data_config['market'])
        
        label_df = D.features(
            stock_list,
            [label_expr],
            start_time=data_config['start_time'],
            end_time=data_config['end_time'],
            freq='day'
        )
        
        label_df.columns = ['LABEL0']
        
        logger.debug(f"  Label rows: {len(label_df)}")
        
        return label_df
    
    def _load_qlib_factors(self, factor_expressions: Dict[str, str]) -> Optional[pd.DataFrame]:
        """Load Qlib-compatible factors."""
        from qlib.data import D
        
        data_config = self.config['data']
        
        try:
            stock_list = D.instruments(data_config['market'])
            
            expressions = list(factor_expressions.values())
            names = list(factor_expressions.keys())
            
            df = D.features(
                stock_list,
                expressions,
                start_time=data_config['start_time'],
                end_time=data_config['end_time'],
                freq='day'
            )
            
            df.columns = names
            return df
        except Exception as e:
            logger.warning(f"Failed to load Qlib factors: {e}")
            return None

    def _normalize_multiindex(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Ensure DataFrame index is MultiIndex(datetime, instrument)."""
        if not isinstance(df.index, pd.MultiIndex):
            return df

        names = list(df.index.names)
        new_names = list(names)
        for i, name in enumerate(names):
            level_vals = df.index.get_level_values(i)
            if name in ("datetime", "date"):
                new_names[i] = "datetime"
            elif name in ("instrument", "stock"):
                new_names[i] = "instrument"
            elif name is None:
                if pd.api.types.is_datetime64_any_dtype(level_vals):
                    new_names[i] = "datetime"
                else:
                    new_names[i] = "instrument"
        if new_names != names:
            df.index = df.index.set_names(new_names)
        actual = list(df.index.names)
        if len(actual) == 2 and actual == ["instrument", "datetime"]:
            df = df.swaplevel()
            df = df.sort_index()
        return df

    def _build_model_selection_fit_index(self, index: pd.Index) -> pd.Index:
        """
        In-sample index for feature-selection/postprocess fitting (train+valid).
        """
        try:
            seg = (self.config.get("dataset") or {}).get("segments") or {}
            train_seg = seg.get("train")
            valid_seg = seg.get("valid")
            if not train_seg or not valid_seg:
                return index
            fit_start = pd.Timestamp(train_seg[0])
            fit_end = pd.Timestamp(valid_seg[1])
        except Exception:
            return index

        if isinstance(index, pd.MultiIndex):
            level = "datetime" if "datetime" in index.names else 0
            dt = pd.to_datetime(index.get_level_values(level))
            mask = (dt >= fit_start) & (dt <= fit_end)
            fit_index = index[mask]
        else:
            dt = pd.to_datetime(index)
            mask = (dt >= fit_start) & (dt <= fit_end)
            fit_index = index[mask]

        if len(fit_index) == 0:
            logger.warning("Model-selection fit index is empty; fallback to full sample")
            return index
        return fit_index

    def _prepare_feature_label_data(
        self,
        factor_expressions: Dict[str, str],
        computed_factors: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build a full-period feature matrix and label series for walk-forward.
        """
        dataset_config = self.config["dataset"]
        label_expr = dataset_config["label"]
        label_df = self._compute_label(label_expr)
        label_df = self._normalize_multiindex(label_df, "label")

        if computed_factors is not None and not computed_factors.empty:
            computed_factors = self._normalize_multiindex(computed_factors, "computed_factors")
            try:
                from .factor_postprocess import FactorPostProcessor, FactorPostprocessConfig
                post_cfg = FactorPostprocessConfig.from_dict(self.config.get("factor_postprocess"))
                if post_cfg.enabled:
                    fit_index = self._build_model_selection_fit_index(computed_factors.index)
                    processor = FactorPostProcessor(post_cfg)
                    computed_factors = processor.process(
                        computed_factors,
                        label_df=label_df,
                        fit_index=fit_index,
                    )
            except Exception as e:
                logger.warning(f"Factor postprocess skipped in walk-forward mode: {e}")

        feature_parts = []
        if computed_factors is not None and not computed_factors.empty:
            feature_parts.append(computed_factors)
        if factor_expressions:
            qlib_factors = self._load_qlib_factors(factor_expressions)
            if qlib_factors is not None and not qlib_factors.empty:
                qlib_factors = self._normalize_multiindex(qlib_factors, "qlib_factors")
                feature_parts.append(qlib_factors)

        if not feature_parts:
            raise ValueError("No features available for walk-forward")

        features_df = pd.concat(feature_parts, axis=1)
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]
        features_df = self._normalize_multiindex(features_df, "features")

        common_index = features_df.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("Feature/label index intersection is empty in walk-forward mode")
        features_df = features_df.loc[common_index]
        label_df = label_df.loc[common_index]

        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        label_col = label_df.columns[0]
        label_s = label_df[label_col].replace([np.inf, -np.inf], np.nan)

        # Cross-sectional rank normalization (same style as existing pipeline).
        dt_level = "datetime" if "datetime" in features_df.index.names else features_df.index.names[0]
        for col in features_df.columns:
            features_df[col] = features_df.groupby(level=dt_level)[col].transform(
                lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0.0
            )
        label_s = label_s.groupby(level=dt_level).transform(
            lambda x: (x.rank(pct=True) - 0.5) if len(x) > 1 else 0.0
        )
        valid = label_s.notna()
        features_df = features_df.loc[valid]
        label_s = label_s.loc[valid]
        return features_df, label_s

    def _load_close_and_benchmark_for_range(
        self,
        start_time: str,
        end_time: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load close matrix for universe and benchmark returns."""
        from qlib.data import D
        from .universe import (
            UniverseConfig,
            apply_liquidity_filters,
            load_historical_universe_mask,
            load_universe,
        )

        market = self.config["data"]["market"]
        benchmark = self.config["backtest"]["backtest"].get("benchmark")
        instruments = D.instruments(market)
        stock_list = D.list_instruments(
            instruments,
            start_time=start_time,
            end_time=end_time,
            as_list=True
        )
        px = D.features(
            stock_list,
            ["$close", "$volume"],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        close_series = px["$close"]
        vol_series = px["$volume"] if "$volume" in px.columns else None
        if isinstance(close_series.index, pd.MultiIndex):
            if "instrument" in close_series.index.names:
                close_mat = close_series.unstack(level="instrument")
                vol_mat = vol_series.unstack(level="instrument") if vol_series is not None else None
            else:
                lvl0 = close_series.index.get_level_values(0)
                close_mat = close_series.unstack(level=1 if pd.api.types.is_datetime64_any_dtype(lvl0) else 0)
                vol_mat = (
                    vol_series.unstack(level=1 if pd.api.types.is_datetime64_any_dtype(lvl0) else 0)
                    if vol_series is not None
                    else None
                )
        else:
            raise ValueError("Close series must be MultiIndex")
        close_mat = close_mat.sort_index().sort_index(axis=1)
        vol_mat = vol_mat.sort_index().sort_index(axis=1) if vol_mat is not None else None

        # Optional universe + liquidity filters from config.
        try:
            u_cfg = UniverseConfig.from_dict(self.config.get("universe"))
            selected = load_universe(u_cfg)
            if selected:
                selected = [s for s in selected if s in close_mat.columns]
                close_mat = close_mat.reindex(columns=selected)
                if vol_mat is not None:
                    vol_mat = vol_mat.reindex(columns=selected)

            hist_mask = load_historical_universe_mask(
                cfg=u_cfg,
                dates=close_mat.index,
                symbols=close_mat.columns,
            )
            if hist_mask is not None:
                close_mat = close_mat.where(hist_mask)
                if vol_mat is not None:
                    vol_mat = vol_mat.where(hist_mask)
                close_mat = close_mat.dropna(axis=1, how="all")
                if vol_mat is not None:
                    vol_mat = vol_mat.reindex(columns=close_mat.columns)

            dollar_volume = close_mat * vol_mat if vol_mat is not None else None
            filtered = apply_liquidity_filters(
                symbols=list(close_mat.columns),
                close_prices=close_mat,
                dollar_volume=dollar_volume,
                min_price=u_cfg.min_price,
                max_price=u_cfg.max_price,
                min_adv=u_cfg.min_avg_dollar_volume or u_cfg.min_adv,
            )
            if filtered:
                close_mat = close_mat.reindex(columns=filtered)
        except Exception as e:
            logger.warning(f"Universe filter skipped due to error: {e}")

        bench_ret = pd.Series(0.0, index=close_mat.index)
        if benchmark:
            try:
                bdf = D.features(
                    [benchmark],
                    ["$close"],
                    start_time=start_time,
                    end_time=end_time,
                    freq="day",
                )
                b = bdf["$close"]
                if isinstance(b.index, pd.MultiIndex):
                    if "instrument" in b.index.names:
                        b = b.unstack(level="instrument").iloc[:, 0]
                    else:
                        lv0 = b.index.get_level_values(0)
                        b = b.unstack(level=1 if pd.api.types.is_datetime64_any_dtype(lv0) else 0).iloc[:, 0]
                bench_ret = b.pct_change().shift(-1).reindex(close_mat.index).fillna(0.0)
            except Exception as e:
                logger.warning(f"Failed to load benchmark returns for walk-forward: {e}")
        return close_mat, bench_ret

    def _fit_predict_lgb(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
    ) -> pd.Series:
        """Fit a lightweight LGB model for one walk-forward window."""
        params = dict((self.config.get("model") or {}).get("params", {}) or {})
        params.pop("early_stopping_round", None)
        n_estimators = int(params.pop("num_boost_round", 300))

        # Normalize common aliases and add robust defaults for walk-forward windows.
        if "loss" in params and "objective" not in params:
            params["objective"] = params.pop("loss")
        else:
            params.pop("loss", None)
        params.setdefault("objective", "regression")
        params.setdefault("n_estimators", n_estimators)
        params.setdefault("verbosity", -1)
        params.setdefault("min_child_samples", 20)
        params.setdefault("feature_fraction", params.pop("colsample_bytree", 0.8))
        params.setdefault("bagging_fraction", params.pop("subsample", 0.8))
        params.setdefault("bagging_freq", 1)
        params.setdefault("num_leaves", 63)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("reg_alpha", params.pop("lambda_l1", 0.0))
        params.setdefault("reg_lambda", params.pop("lambda_l2", 1.0))

        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
            model.fit(x_train, y_train)
            # Degenerate tree ensembles frequently happen in low-signal windows.
            # Fallback to a linear baseline if no feature splits are learned.
            if hasattr(model, "feature_importances_") and float(np.sum(model.feature_importances_)) <= 0.0:
                raise ValueError("Degenerate LightGBM model (no learned splits)")
            pred = model.predict(x_test)
        except Exception as e:
            logger.warning(f"LightGBM fit failed in walk-forward window, fallback to linear model: {e}")
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(x_train.values, y_train.values)
            pred = model.predict(x_test.values)
        return pd.Series(pred, index=x_test.index, name="pred")

    def _run_walk_forward_backtest(
        self,
        factor_expressions: Dict[str, str],
        computed_factors: Optional[pd.DataFrame],
        exp_name: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run rolling walk-forward backtest with OOS stitching."""
        from .walk_forward import WalkForwardConfig, WalkForwardEngine
        from .enhanced_portfolio import EnhancedPortfolioBacktester, EnhancedPortfolioConfig
        from .cost_model import CostModelConfig
        from .metrics import compute_backtest_metrics

        features_df, label_s = self._prepare_feature_label_data(factor_expressions, computed_factors)
        idx_dates = features_df.index.get_level_values("datetime") if "datetime" in features_df.index.names else features_df.index.get_level_values(0)
        start_time = str(pd.Timestamp(idx_dates.min()).date())
        end_time = str(pd.Timestamp(idx_dates.max()).date())
        close_mat, bench_ret = self._load_close_and_benchmark_for_range(start_time, end_time)

        wf_cfg = WalkForwardConfig.from_dict(self.config.get("walk_forward"))
        wf_engine = WalkForwardEngine(wf_cfg)
        port_cfg = EnhancedPortfolioConfig.from_dict((self.config.get("backtest", {}) or {}).get("enhanced", {}))
        cost_cfg = CostModelConfig.from_dict(self.config.get("cost_model"))

        window_details = []
        oos_port_returns = []
        oos_turnover = []
        oos_costs = []

        def _evaluate_window(pred: pd.Series, window) -> Dict:
            # Evaluate only on window test range.
            test_close = close_mat.loc[(close_mat.index >= window.test_start) & (close_mat.index <= window.test_end)]
            test_bench = bench_ret.loc[(bench_ret.index >= window.test_start) & (bench_ret.index <= window.test_end)]
            portfolio = EnhancedPortfolioBacktester(port_cfg)
            portfolio.set_cost_model(cost_cfg)
            daily_df, m = portfolio.run(signal=pred, close_df=test_close, benchmark_returns=test_bench)
            return {"daily": daily_df, "metrics": m, "oos_returns": daily_df["portfolio_return"]}

        wf_result = wf_engine.run(
            features=features_df,
            label=label_s,
            fit_predict_fn=self._fit_predict_lgb,
            evaluate_fn=_evaluate_window,
        )

        for item in wf_result.get("windows", []):
            w = item["window"]
            daily = item["daily"]
            m = item["metrics"]
            window_details.append({
                "idx": w.idx,
                "train_start": str(w.train_start.date()),
                "train_end": str(w.train_end.date()),
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "metrics": m,
            })
            oos_port_returns.append(daily["portfolio_return"])
            oos_turnover.append(daily.get("turnover", pd.Series(dtype=float)))
            oos_costs.append(daily.get("cost", pd.Series(dtype=float)))

        if not oos_port_returns:
            return {}

        oos_port = pd.concat(oos_port_returns).sort_index()
        oos_port = oos_port[~oos_port.index.duplicated(keep="first")]
        oos_turn = pd.concat(oos_turnover).sort_index() if oos_turnover else pd.Series(dtype=float)
        oos_cost = pd.concat(oos_costs).sort_index() if oos_costs else pd.Series(dtype=float)
        oos_bench = bench_ret.reindex(oos_port.index).fillna(0.0)

        suite = compute_backtest_metrics(
            strategy_returns=oos_port,
            benchmark_returns=oos_bench,
            turnover=oos_turn,
            costs=oos_cost,
            num_positions=None,
        ).to_dict()
        metrics = {
            "annualized_return": suite.get("cagr", 0.0),
            "information_ratio": suite.get("information_ratio", 0.0),
            "sharpe_ratio": suite.get("sharpe_ratio", 0.0),
            "max_drawdown": suite.get("max_drawdown", 0.0),
            "calmar_ratio": suite.get("calmar_ratio", 0.0),
            "avg_turnover": float(oos_turn.mean()) if len(oos_turn) > 0 else 0.0,
            "walk_forward_windows": float(len(window_details)),
        }
        # Include full suite for downstream analysis.
        metrics.update({f"wf_{k}": v for k, v in suite.items()})

        output_dir = Path(self.config["experiment"].get("output_dir", "./backtest_v2_results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = output_name or exp_name
        wf_csv = output_dir / f"{file_prefix}_walk_forward_oos.csv"
        oos_df = pd.DataFrame({
            "portfolio_return": oos_port,
            "benchmark_return": oos_bench.reindex(oos_port.index),
        })
        oos_df["excess_return"] = oos_df["portfolio_return"] - oos_df["benchmark_return"]
        oos_df["cumulative_oos_return"] = (1.0 + oos_df["portfolio_return"]).cumprod() - 1.0
        oos_df.to_csv(wf_csv, index=True)
        self._save_daily_excess_csv(oos_df["excess_return"], file_prefix=file_prefix)

        wf_json = output_dir / f"{file_prefix}_walk_forward_windows.json"
        with open(wf_json, "w", encoding="utf-8") as f:
            json.dump(window_details, f, ensure_ascii=False, indent=2)
        logger.info(f"Walk-forward outputs saved: {wf_csv}, {wf_json}")
        return metrics
    
    def _train_and_backtest(self, dataset, exp_name: str, rec_name: str, output_name: Optional[str] = None) -> Dict:
        """Train model and run backtest."""
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
        
        model_config = self.config['model']
        backtest_config = self.config['backtest']['backtest']
        strategy_config = self.config['backtest']['strategy']
        
        metrics = {}
        
        with R.start(experiment_name=exp_name, recorder_name=rec_name):
            # Train model
            train_start = time.time()
            
            if model_config['type'] == 'lgb':
                model = LGBModel(**model_config['params'])
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
            
            model.fit(dataset)
            print(f"[4/4] Train LightGBM done ({time.time()-train_start:.1f}s)")
            
            # Generate prediction
            pred = model.predict(dataset)
            logger.debug(f"  Pred shape: {pred.shape}")
            metrics.update(self._compute_segment_ic_metrics(pred))
            
            # Save prediction
            sr = SignalRecord(recorder=R.get_recorder(), model=model, dataset=dataset)
            sr.generate()
            
            # Compute IC metrics
            try:
                sar = SigAnaRecord(recorder=R.get_recorder(), ana_long_short=False, ann_scaler=252)
                sar.generate()
                
                recorder = R.get_recorder()
                try:
                    ic_series = recorder.load_object("sig_analysis/ic.pkl")
                    ric_series = recorder.load_object("sig_analysis/ric.pkl")
                    
                    if isinstance(ic_series, pd.Series) and len(ic_series) > 0:
                        metrics['IC'] = float(ic_series.mean())
                        metrics['ICIR'] = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
                    
                    if isinstance(ric_series, pd.Series) and len(ric_series) > 0:
                        metrics['Rank IC'] = float(ric_series.mean())
                        metrics['Rank ICIR'] = float(ric_series.mean() / ric_series.std()) if ric_series.std() > 0 else 0.0
                    
                    print(f"  IC={metrics.get('IC', 0):.6f}, ICIR={metrics.get('ICIR', 0):.6f}, "
                          f"Rank IC={metrics.get('Rank IC', 0):.6f}, Rank ICIR={metrics.get('Rank ICIR', 0):.6f}")
                except Exception as e:
                    logger.warning(f"Could not read IC result: {e}")
            except Exception as e:
                logger.warning(f"IC analysis failed: {e}")
            # Portfolio backtest
            try:
                bt_start = time.time()
                bt_mode = str(self.config.get('backtest', {}).get('mode', 'qlib')).lower()
                file_prefix = output_name if output_name else exp_name

                if bt_mode == 'enhanced':
                    enhanced = self._run_enhanced_portfolio_backtest(pred, file_prefix=file_prefix)
                    metrics.update(enhanced)
                    print(f"  Enhanced portfolio backtest done ({time.time()-bt_start:.1f}s)")
                else:
                    qlib_metrics = self._run_qlib_portfolio_backtest(
                        pred=pred,
                        strategy_config=strategy_config,
                        backtest_config=backtest_config,
                        file_prefix=file_prefix,
                    )
                    metrics.update(qlib_metrics)
                    print(f"  Portfolio backtest done ({time.time()-bt_start:.1f}s)")

            except Exception as e:
                logger.warning(f"Portfolio backtest failed: {e}")
                import traceback
                traceback.print_exc()
        
        return metrics

    def _compute_segment_ic_metrics(self, pred: pd.Series) -> Dict[str, float]:
        """Compute IC/RankIC per segment (train/valid/test)."""
        try:
            label_expr = self.config["dataset"]["label"]
            label_df = self._compute_label(label_expr)
            if isinstance(label_df, pd.DataFrame):
                label_df = self._normalize_multiindex(label_df, "segment_ic_label")
            if isinstance(label_df, pd.DataFrame) and not label_df.empty:
                label_s = label_df.iloc[:, 0]
            elif isinstance(label_df, pd.Series):
                label_s = label_df
            else:
                return {}
            label_s = label_s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

            pred_s = pred.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if isinstance(pred_s.index, pd.MultiIndex):
                pred_df = self._normalize_multiindex(pred_s.to_frame(name="pred"), "segment_ic_pred")
                pred_s = pred_df["pred"]
            common_index = pred_s.index.intersection(label_s.index)
            if len(common_index) == 0:
                return {}
            pred_s = pred_s.loc[common_index]
            label_s = label_s.loc[common_index]

            dt_level = "datetime" if isinstance(pred_s.index, pd.MultiIndex) and "datetime" in pred_s.index.names else 0
            dt_index = pd.to_datetime(pred_s.index.get_level_values(dt_level))
            segments = (self.config.get("dataset") or {}).get("segments") or {}

            out: Dict[str, float] = {}
            for seg_name in ("train", "valid", "test"):
                seg = segments.get(seg_name)
                if not seg:
                    continue
                seg_start = pd.Timestamp(seg[0])
                seg_end = pd.Timestamp(seg[1])
                mask = (dt_index >= seg_start) & (dt_index <= seg_end)
                if int(mask.sum()) == 0:
                    continue
                seg_pred = pred_s.loc[mask]
                seg_label = label_s.loc[mask]
                ic_mean, ic_ir = self._cross_sectional_ic_stats(seg_pred, seg_label, method="pearson")
                ric_mean, ric_ir = self._cross_sectional_ic_stats(seg_pred, seg_label, method="spearman")
                out[f"{seg_name}_IC"] = ic_mean
                out[f"{seg_name}_ICIR"] = ic_ir
                out[f"{seg_name}_Rank IC"] = ric_mean
                out[f"{seg_name}_Rank ICIR"] = ric_ir
            return out
        except Exception as e:
            logger.warning(f"Segment IC calculation failed: {e}")
            return {}

    @staticmethod
    def _cross_sectional_ic_stats(
        pred_s: pd.Series,
        label_s: pd.Series,
        method: str = "pearson",
    ) -> Tuple[float, float]:
        if pred_s.empty or label_s.empty:
            return 0.0, 0.0
        df = pd.DataFrame({"pred": pred_s, "label": label_s}).dropna()
        if df.empty:
            return 0.0, 0.0
        dt_level = "datetime" if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names else 0
        ic_series = df.groupby(level=dt_level).apply(
            lambda x: x["pred"].corr(x["label"], method=method) if len(x) > 1 else np.nan
        )
        ic_series = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
        if ic_series.empty:
            return 0.0, 0.0
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        return ic_mean, ic_ir

    def _run_qlib_portfolio_backtest(
        self,
        pred: pd.Series,
        strategy_config: Dict[str, Any],
        backtest_config: Dict[str, Any],
        file_prefix: str,
    ) -> Dict[str, float]:
        """Run legacy Qlib TopkDropout backtest and extract strategy metrics."""
        from qlib.data import D
        from qlib.backtest import backtest as qlib_backtest
        from qlib.contrib.evaluate import risk_analysis

        metrics: Dict[str, float] = {}
        market = self.config['data']['market']
        instruments = D.instruments(market)
        stock_list = D.list_instruments(
            instruments,
            start_time=backtest_config['start_time'],
            end_time=backtest_config['end_time'],
            as_list=True
        )
        logger.debug(f"  Stock count: {len(stock_list)}")
        if len(stock_list) < 10:
            logger.warning(f"Stock pool too small ({len(stock_list)}), results may be unreliable")

        # Filter invalid price signals
        try:
            price_data = D.features(
                stock_list,
                ['$close'],
                start_time=backtest_config['start_time'],
                end_time=backtest_config['end_time'],
                freq='day'
            )
            invalid_mask = (price_data['$close'] == 0) | (price_data['$close'].isna())
            if int(invalid_mask.sum()) > 0 and isinstance(pred, pd.Series):
                invalid_indices = invalid_mask[invalid_mask].index
                invalid_set = set((dt, inst) for inst, dt in invalid_indices)
                filtered = 0
                for idx in pred.index:
                    if idx in invalid_set:
                        pred.loc[idx] = np.nan
                        filtered += 1
                if filtered > 0:
                    logger.debug(f"  Filtered {filtered} invalid price signals")
        except Exception as filter_err:
            logger.warning(f"Price filter failed: {filter_err}")

        portfolio_metric_dict, _ = qlib_backtest(
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                    "verbose": False,
                    "indicator_config": {"show_indicator": False}
                }
            },
            strategy={
                "class": strategy_config['class'],
                "module_path": strategy_config['module_path'],
                "kwargs": {
                    "signal": pred,
                    "topk": strategy_config['kwargs']['topk'],
                    "n_drop": strategy_config['kwargs']['n_drop']
                }
            },
            start_time=backtest_config['start_time'],
            end_time=backtest_config['end_time'],
            account=backtest_config['account'],
            benchmark=backtest_config['benchmark'],
            exchange_kwargs={
                "codes": stock_list,
                **backtest_config['exchange_kwargs']
            }
        )

        if not portfolio_metric_dict or "1day" not in portfolio_metric_dict:
            return metrics

        report_df, _ = portfolio_metric_dict["1day"]
        if not isinstance(report_df, pd.DataFrame) or 'return' not in report_df.columns:
            return metrics

        portfolio_return = report_df['return'].replace([np.inf, -np.inf], np.nan).fillna(0)
        bench_return = report_df['bench'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'bench' in report_df.columns else 0
        cost = report_df['cost'].replace([np.inf, -np.inf], np.nan).fillna(0) if 'cost' in report_df.columns else 0
        excess_return_with_cost = (portfolio_return - bench_return - cost).dropna()
        if excess_return_with_cost.empty:
            return metrics

        self._save_daily_excess_csv(excess_return_with_cost, file_prefix=file_prefix)

        analysis = risk_analysis(excess_return_with_cost)
        if isinstance(analysis, pd.DataFrame):
            analysis = analysis['risk'] if 'risk' in analysis.columns else analysis.iloc[:, 0]

        ann_ret = float(analysis.get('annualized_return', 0))
        info_ratio = float(analysis.get('information_ratio', 0))
        max_dd = float(analysis.get('max_drawdown', 0))

        if np.isfinite(ann_ret):
            metrics['annualized_return'] = ann_ret
        if np.isfinite(info_ratio):
            metrics['information_ratio'] = info_ratio
        if np.isfinite(max_dd):
            metrics['max_drawdown'] = max_dd
        if max_dd != 0 and np.isfinite(ann_ret):
            calmar = ann_ret / abs(max_dd)
            if np.isfinite(calmar):
                metrics['calmar_ratio'] = calmar
        return metrics

    def _run_enhanced_portfolio_backtest(self, pred: pd.Series, file_prefix: str) -> Dict[str, float]:
        """Run enhanced non-HFT portfolio backtest."""
        from .enhanced_portfolio import EnhancedPortfolioBacktester, EnhancedPortfolioConfig
        from .cost_model import CostModelConfig

        backtest_config = self.config['backtest']['backtest']
        close_mat, bench_ret = self._load_close_and_benchmark_for_range(
            start_time=backtest_config['start_time'],
            end_time=backtest_config['end_time'],
        )
        if close_mat is None or close_mat.empty:
            logger.warning("No price matrix available for enhanced backtest")
            return {}

        cfg = EnhancedPortfolioConfig.from_dict(
            (self.config.get('backtest', {}) or {}).get('enhanced', {}) or {}
        )
        engine = EnhancedPortfolioBacktester(cfg)
        engine.set_cost_model(CostModelConfig.from_dict(self.config.get("cost_model")))
        daily_df, metrics = engine.run(signal=pred, close_df=close_mat, benchmark_returns=bench_ret)

        # Persist full daily report and cumulative excess file for plotting.
        output_dir = Path(self.config['experiment'].get('output_dir', './backtest_v2_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        daily_path = output_dir / f"{file_prefix}_enhanced_daily.csv"
        daily_df.to_csv(daily_path, index=True)
        self._save_daily_excess_csv(daily_df['excess_return'], file_prefix=file_prefix)
        logger.info(f"Enhanced daily report saved: {daily_path}")
        return metrics

    def _save_daily_excess_csv(self, excess_return: pd.Series, file_prefix: str):
        """Save daily/cumulative excess return to CSV for downstream plotting."""
        try:
            output_dir = Path(self.config['experiment'].get('output_dir', './backtest_v2_results'))
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{file_prefix}_cumulative_excess.csv"
            save_df = pd.DataFrame({
                'daily_excess_return': excess_return.astype(float).fillna(0.0)
            })
            save_df['cumulative_excess_return'] = save_df['daily_excess_return'].cumsum()
            save_df.index.name = 'date'
            save_df.to_csv(csv_path)
            logger.debug(f"  Daily excess return saved: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save daily excess CSV: {e}")
    
    def _print_results(self, metrics: Dict, total_time: float):
        """Print result summary."""
        def _f(val, fmt='.6f'):
            return format(val, fmt) if isinstance(val, (int, float)) else 'N/A'

        print(f"\n{'='*50}")
        print("Backtest Results")
        print(f"{'='*50}")
        print("[IC Metrics]")
        print(f"  IC: {_f(metrics.get('IC'))}  ICIR: {_f(metrics.get('ICIR'))}")
        print(f"  Rank IC: {_f(metrics.get('Rank IC'))}  Rank ICIR: {_f(metrics.get('Rank ICIR'))}")
        for seg in ("train", "valid", "test"):
            seg_ic = metrics.get(f"{seg}_IC")
            if seg_ic is None:
                continue
            print(
                f"  {seg.upper()} IC: {_f(seg_ic)}  {seg.upper()} ICIR: {_f(metrics.get(f'{seg}_ICIR'))}  "
                f"{seg.upper()} Rank IC: {_f(metrics.get(f'{seg}_Rank IC'))}  "
                f"{seg.upper()} Rank ICIR: {_f(metrics.get(f'{seg}_Rank ICIR'))}"
            )
        print("[Strategy Metrics]")
        print(f"  Ann. Return: {_f(metrics.get('annualized_return'), '.4f')}  Max DD: {_f(metrics.get('max_drawdown'), '.4f')}")
        print(f"  Info Ratio: {_f(metrics.get('information_ratio'), '.4f')}  Calmar: {_f(metrics.get('calmar_ratio'), '.4f')}")
        if metrics.get('sharpe_ratio') is not None:
            print(f"  Sharpe: {_f(metrics.get('sharpe_ratio'), '.4f')}  Avg Turnover: {_f(metrics.get('avg_turnover'), '.4f')}")
        print(f"Total time: {total_time:.1f}s")
        print(f"{'='*50}")
    
    def _save_results(self, metrics: Dict, exp_name: str, 
                     factor_source: str, num_factors: int, elapsed: float,
                     output_name: Optional[str] = None):
        """Save results."""
        output_dir = Path(self.config['experiment'].get('output_dir', './backtest_v2_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_name:
            output_file = f"{output_name}_backtest_metrics.json"
        else:
            output_file = self.config['experiment']['output_metrics_file']
        output_path = output_dir / output_file
        
        result_data = {
            "experiment_name": exp_name,
            "factor_source": factor_source,
            "num_factors": num_factors,
            "metrics": metrics,
            "config_full": self.config,
            "config": {
                "data_range": f"{self.config['data']['start_time']} ~ {self.config['data']['end_time']}",
                "test_range": f"{self.config['dataset']['segments']['test'][0]} ~ {self.config['dataset']['segments']['test'][1]}",
                "backtest_range": f"{self.config['backtest']['backtest']['start_time']} ~ {self.config['backtest']['backtest']['end_time']}",
                "market": self.config['data']['market'],
                "benchmark": self.config['backtest']['backtest']['benchmark']
            },
            "elapsed_seconds": elapsed
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved: {output_path}")
        summary_file = output_dir / "batch_summary.json"
        summary_data = []
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except:
                summary_data = []
        
        ann_ret = metrics.get('annualized_return')
        mdd = metrics.get('max_drawdown')
        calmar_ratio = None
        if ann_ret is not None and mdd is not None and mdd != 0:
            calmar_ratio = ann_ret / abs(mdd)
        
        summary_entry = {
            "name": output_name or exp_name,
            "num_factors": num_factors,
            "IC": metrics.get('IC'),
            "ICIR": metrics.get('ICIR'),
            "Rank_IC": metrics.get('Rank IC'),
            "Rank_ICIR": metrics.get('Rank ICIR'),
            "train_IC": metrics.get("train_IC"),
            "valid_IC": metrics.get("valid_IC"),
            "test_IC": metrics.get("test_IC"),
            "annualized_return": ann_ret,
            "information_ratio": metrics.get('information_ratio'),
            "sharpe_ratio": metrics.get('sharpe_ratio'),
            "max_drawdown": mdd,
            "calmar_ratio": calmar_ratio,
            "avg_turnover": metrics.get('avg_turnover'),
            "elapsed_seconds": elapsed
        }
        summary_data.append(summary_entry)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Appended to summary: {summary_file}")
