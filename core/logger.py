"""
Centralized logging system for the parking optimization project.
Provides structured logging with performance monitoring and configurable outputs.
"""

import functools
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, ClassVar, Dict, Optional

from .config import get_config


class PerformanceLogger:
    """Logger with built-in performance monitoring"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.config = get_config()
        self._setup_logger()

    def _setup_logger(self):
        """Configure logger based on settings"""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set level
        level = getattr(logging, self.config.output.log_level.upper())
        self.logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if enabled)
        if self.config.output.log_to_file:
            # Try to get run-specific log file if run manager is active
            log_file = self._get_log_file_path()

            # Rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.output.max_log_file_size_mb * 1024 * 1024,
                backupCount=5,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _get_log_file_path(self):
        """Get appropriate log file path - per-run if available, otherwise global"""
        try:
            # Try to import run manager and get current run path
            from core.run_manager import get_run_manager

            run_manager = get_run_manager()
            if hasattr(run_manager, "current_run_dir") and run_manager.current_run_dir:
                # Create run-specific log file
                log_file = run_manager.get_run_path("logs") / f"{self.logger.name}.log"
                return log_file
        except (ImportError, AttributeError, RuntimeError):
            # Fall back to global log file if run manager isn't available
            pass

        # Default global log file
        return (
            self.config.output.logs_dir
            / f"{datetime.now().strftime('%Y%m%d')}_parking_optimization.log"
        )

    def reconfigure_for_run(self, run_directory=None):
        """Reconfigure logger to use run-specific log file if a run is active"""
        if not self.config.output.log_to_file:
            return

        # Remove existing file handlers
        handlers_to_remove = [
            h
            for h in self.logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        for handler in handlers_to_remove:
            self.logger.removeHandler(handler)

        # Add new run-specific file handler
        if run_directory:
            log_file = run_directory / "logs" / f"{self.logger.name}.log"
        else:
            log_file = self._get_log_file_path()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Create logs directory if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Add new file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.output.max_log_file_size_mb * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context"""
        if kwargs:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"
        self.logger.log(level, message)

    @contextmanager
    def performance_context(self, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        self.info(f"Starting {operation}")

        try:
            yield
            duration = time.time() - start_time
            self.info(f"Completed {operation}", duration_seconds=f"{duration:.3f}")
        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"Failed {operation}", duration_seconds=f"{duration:.3f}", error=str(e)
            )
            raise

    def time_function(self, func_name: Optional[str] = None):
        """Decorator to time function execution"""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                operation = func_name or f"{func.__module__}.{func.__name__}"
                with self.performance_context(operation):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class ModuleLogger:
    """Logger factory for different modules"""

    _loggers: ClassVar[Dict[str, PerformanceLogger]] = {}

    @classmethod
    def get_logger(cls, module_name: str) -> PerformanceLogger:
        """Get or create logger for a module"""
        if module_name not in cls._loggers:
            cls._loggers[module_name] = PerformanceLogger(module_name)
        return cls._loggers[module_name]


# Convenience functions for getting loggers
def get_logger(module_name: str) -> PerformanceLogger:
    """Get logger for a specific module"""
    return ModuleLogger.get_logger(module_name)


# Pre-configured loggers for main modules
main_logger = get_logger("main")
simulation_logger = get_logger("simulation")
pricing_logger = get_logger("pricing")
routing_logger = get_logger("routing")
traffic_logger = get_logger("traffic")
analysis_logger = get_logger("analysis")


# Performance monitoring decorators
def time_it(func_name: Optional[str] = None):
    """Decorator to time any function"""

    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__)
        return logger.time_function(func_name)(func)

    return decorator


def log_entry_exit(logger_name: Optional[str] = None):
    """Decorator to log function entry and exit"""

    def decorator(func: Callable) -> Callable:
        logger = get_logger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


class MetricsCollector:
    """Collect and log performance metrics"""

    def __init__(self):
        self.metrics = {}
        self.logger = get_logger("metrics")

    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record a performance metric"""
        self.metrics[name] = {"value": value, "unit": unit, "timestamp": datetime.now()}
        self.logger.info(f"Metric recorded: {name}={value} {unit}")

    def record_duration(self, name: str, duration_seconds: float):
        """Record a duration metric"""
        self.record_metric(name, duration_seconds, "seconds")

    def record_count(self, name: str, count: int):
        """Record a count metric"""
        self.record_metric(name, count, "items")

    def record_rate(self, name: str, rate: float, unit: str = "per_second"):
        """Record a rate metric"""
        self.record_metric(name, rate, unit)

    def get_metrics_summary(self) -> dict:
        """Get summary of all recorded metrics"""
        return {"total_metrics": len(self.metrics), "metrics": self.metrics}

    def log_summary(self):
        """Log summary of all metrics"""
        summary = self.get_metrics_summary()
        self.logger.info("Performance metrics summary", **summary)


# Global metrics collector
metrics = MetricsCollector()


def reconfigure_all_loggers_for_run(run_directory):
    """Reconfigure all active loggers to use run-specific log files"""
    for logger in ModuleLogger._loggers.values():
        logger.reconfigure_for_run(run_directory)


def setup_logging():
    """Initialize logging system"""
    config = get_config()

    # Create logs directory
    config.output.logs_dir.mkdir(exist_ok=True, parents=True)

    # Log startup information
    main_logger.info("Parking optimization system starting")
    main_logger.info(
        "Configuration loaded",
        api_keys_available=config.has_api_keys,
        log_level=config.output.log_level,
    )

    # Log any configuration warnings
    warnings = config.validate()
    for warning in warnings:
        main_logger.warning(f"Configuration warning: {warning}")


# Ensure logging is set up when module is imported
setup_logging()
