"""
Real-time monitoring of GPU, CPU, and Memory usage.
Runs in a background daemon thread and periodically logs resource stats.
"""

import os
import time
import threading
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ResourceMonitor")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PUPPLE = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _format_bytes(bytes_val: float) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


class ResourceMonitor:
    """
    Background monitor for GPU, CPU, and RAM usage.

    Parameters
    ----------
    interval : float
        Logging interval in seconds. Default 5.0.
    log_to_file : str or None
        If provided, also write stats to this CSV file.
    """

    def __init__(self, interval: float = 5.0, log_to_file: str = None):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self._log_file = log_to_file
        self._file_handle = None
        self._current_workload = ""

        # --- GPU ---
        self._gpu_available = False
        self._gpu_count = 0
        self._nvml_handle = None

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                if self._gpu_count > 0:
                    self._gpu_available = True
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                else:
                    pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
        else:
            logger.warning("pynvml not installed — GPU monitoring disabled.")

        logger.info(
            f"{PUPPLE}ResourceMonitor initialized: "
            f"GPU={'available' if self._gpu_available else 'unavailable'} "
            f"(count={self._gpu_count}), "
            f"CPU={'available' if HAS_PSUTIL else 'unavailable'}{RESET}"
        )

    # ------------------------------------------------------------------
    #  GPU stats
    # ------------------------------------------------------------------
    def _get_gpu_stats(self) -> dict:
        """Return GPU stats dict or empty dict on failure."""
        if not self._gpu_available:
            return {}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return {
                "GPU_Util_Pct": util.gpu,
                "GPU_Mem_Pct": (mem.used / mem.total) * 100 if mem.total > 0 else 0,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    #  CPU / Memory stats
    # ------------------------------------------------------------------
    def _get_cpu_mem_stats(self) -> dict:
        """Return CPU + RAM stats dict or empty dict on failure."""
        if not HAS_PSUTIL:
            return {}
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            return {
                "CPU_Util_Pct": cpu,
                "RAM_Util_Pct": mem.percent,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    #  Logging helpers
    # ------------------------------------------------------------------
    def _log_row(self, elapsed_s: float):
        gpu = self._get_gpu_stats()
        cpu = self._get_cpu_mem_stats()
        stats = {**gpu, **cpu}

        if not stats:
            return

        # elapsed_str = f"{elapsed_s:.1f}s".rjust(8)

        # parts = [f"[{CYAN}{elapsed_str}{RESET}]"]

        # if gpu:
        #     parts.append(
        #         f"{GREEN}GPU:{RESET} "
        #         f"{gpu['gpu_util_pct']:5.1f}% util | "
        #         f"mem {gpu['gpu_mem_used_gb']:.1f}/{gpu['gpu_mem_total_gb']:.1f} GB "
        #         f"({gpu['gpu_mem_pct']:.1f}%) | "
        #         f"temp {gpu['gpu_temp_c']}°C"
        #     )

        # if cpu:
        #     parts.append(
        #         f"{YELLOW}CPU:{RESET} "
        #         f"{cpu['cpu_util_pct']:5.1f}% | "
        #         f"{BLUE}RAM:{RESET} "
        #         f"{cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.1f} GB "
        #         f"({cpu['ram_pct']:.1f}%)"
        #     )

        # print("  ".join(parts))

        # Write CSV header on first row
        if self._file_handle is not None:
            self._write_csv(stats, elapsed_s)

    def _write_csv(self, stats: dict, elapsed_s: float):
        """Append one row to the CSV log file."""
        try:
            if not hasattr(self, '_csv_header_written'):
                keys = ["elapsed_s", "workload"] + list(stats.keys())
                self._file_handle.write(",".join(keys) + "\n")
                self._csv_header_written = True
            row = [f"{elapsed_s:.2f}", self._current_workload] + [f"{stats.get(k, '')}" for k in stats.keys()]
            self._file_handle.write(",".join(row) + "\n")
            self._file_handle.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Background loop
    # ------------------------------------------------------------------
    def _monitor_loop(self):
        """Main loop of the monitor thread."""
        self._start_time = time.time()
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            self._log_row(elapsed)
            self._stop_event.wait(self.interval)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def start(self):
        """Start the background monitoring thread."""
        if self._thread is not None:
            logger.warning("Monitor already running.")
            return

        if os.path.isfile(self._log_file):
            os.remove(self._log_file)

        self._file_handle = open(self._log_file, "w")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._thread = None

        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            logger.info(f"{PUPPLE}ResourceMonitor stopped — log saved.{RESET}")

        # Also print a final snapshot
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._log_row(elapsed)

        if self._gpu_available and HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def set_workload(self, label: str):
        """Set the current workload label shown in CSV rows."""
        self._current_workload = label

    def get_snapshot(self) -> dict:
        """Return current resource snapshot (non-blocking, for programmatic use)."""
        gpu = self._get_gpu_stats()
        cpu = self._get_cpu_mem_stats()
        return {**gpu, **cpu}
