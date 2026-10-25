"""Resource monitor: CPU util, server RSS, GPU util/power/mem (nvidia-smi), CPU package energy (RAPL).

All sources are optional; whatever is unavailable is reported as null (and should be
stated as a limitation in the report instead of being estimated silently).
"""
import glob
import os
import platform
import shutil
import subprocess
import threading
import time

import numpy as np
import psutil


def _read_int(p):
    try:
        with open(p) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


class Monitor:
    def __init__(self, interval=0.25, pids=None):
        self.interval, self.pids = interval, list(pids or [])
        self._stop, self._thr = threading.Event(), None
        self.rows = []  # (t, cpu%, gpu_util, gpu_power_w, gpu_mem_mb, rss_mb)
        self.has_gpu = shutil.which("nvidia-smi") is not None
        self.rapl = []
        for p in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            if os.path.basename(p).count(":") == 1 and _read_int(p + "/energy_uj") is not None:
                self.rapl.append((p + "/energy_uj", _read_int(p + "/max_energy_range_uj") or 2**40))

    def _gpu(self):
        if not self.has_gpu:
            return None, None, None
        try:
            o = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            u, p, m = [], [], []
            for line in o.strip().splitlines():
                a = [_f(x) for x in line.split(",")]
                u.append(a[0]); p.append(a[1]); m.append(a[2])
            if any(x is None for x in p + u + m):
                return None, None, None
            return max(u), sum(p), sum(m)
        except Exception:
            return None, None, None

    def _rss(self):
        tot = 0
        for pid in self.pids:
            try:
                pr = psutil.Process(pid)
                tot += pr.memory_info().rss
                for c in pr.children(recursive=True):
                    tot += c.memory_info().rss
            except Exception:
                pass
        return tot / 2**20 if self.pids else None

    def _loop(self):
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            g = self._gpu()
            self.rows.append((time.time(), psutil.cpu_percent(None), *g, self._rss()))
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.time()
        self.e0 = [_read_int(p) for p, _ in self.rapl]
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thr.join()
        self.t1 = time.time()
        self.e1 = [_read_int(p) for p, _ in self.rapl]

    def summary(self):
        dur = self.t1 - self.t0
        r = np.array([[np.nan if v is None else v for v in row] for row in self.rows], dtype=float) \
            if self.rows else np.zeros((0, 6))

        def stat(col, fn):
            if not len(r) or np.all(np.isnan(r[:, col])):
                return None
            return float(fn(r[~np.isnan(r[:, col]), col]))
        cpu_e = None
        if self.rapl and all(v is not None for v in self.e0 + self.e1):
            cpu_e = 0.0
            for (p, mx), a, b in zip(self.rapl, self.e0, self.e1):
                cpu_e += ((b - a) % mx) / 1e6      # microjoules -> J, wrap-safe
        gpu_e = None
        if len(r) > 1 and not np.all(np.isnan(r[:, 3])):
            m = ~np.isnan(r[:, 3])
            t, p = r[m, 0], r[m, 3]
            gpu_e = float(np.sum((t[1:] - t[:-1]) * (p[1:] + p[:-1]) / 2)) if m.sum() > 1 else None
        return dict(duration_s=dur, cpu_util_mean=stat(1, np.mean), cpu_util_max=stat(1, np.max),
                    gpu_util_mean=stat(2, np.mean), gpu_power_mean_w=stat(3, np.mean),
                    gpu_mem_max_mb=stat(4, np.max), server_rss_max_mb=stat(5, np.max),
                    cpu_energy_j=cpu_e, gpu_energy_j=gpu_e, n_samples=len(r))


def capture_env(extra=None):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    env = dict(platform=platform.platform(), python=platform.python_version(), cpu=cpu,
               logical_cores=os.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
               gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
               git_rev=sh("git rev-parse --short HEAD"), hostname=platform.node())
    env.update(extra or {})
    return env
