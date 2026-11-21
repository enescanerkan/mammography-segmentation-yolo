"""Session protocol of the runtime benchmark: CPU pinning, the job list and its rotating order, the GPU guard.

Only the standard library is imported at module level: :func:`prepare_process` pins the process to its cores and
sets the thread variables *before* numpy, OpenCV and torch are imported, so that every thread they start inherits
the pinning (``os.sched_setaffinity`` acts on the calling thread and the threads created after it).

* **CPU cores.** The ``threads`` physical cores with the lowest load over a short ``/proc/stat`` sample (load of a
  core = sum over its SMT siblings), one logical CPU of each; the process is pinned to them for the whole session
  and runs exactly as many threads as it has cores (explicit cores set the thread count; fewer free cores lower it).
* **GPU.** A GPU index is the one nvidia-smi shows (``CUDA_DEVICE_ORDER=PCI_BUS_ID``); the guard watches the card
  torch actually uses and refuses a ``CUDA_VISIBLE_DEVICES`` that names another card for nvidia-smi.
* **Order.** A fixed random base order of the configurations (``numpy.random.default_rng(seed).permutation``),
  rotated by one position per job, so whatever else the machine does hits every configuration alike. Jobs are the
  warm-up pairs, then every pair once per pass.
* **GPU guard.** The compute processes on the session's card (``nvidia-smi --query-compute-apps``) are checked at
  the start, every ``card_check_every`` jobs and at the end; any process other than this one and the listed
  persistent ones (``allowed_pids``) makes the measurement invalid (:class:`InvalidMeasurement`, exit code 4 on the
  command line). So does a card that stays busy at the start although no foreign process is listed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
#: Modules whose thread pools would no longer follow the pinning if imported before :func:`prepare_process`.
THREADED_MODULES = ("numpy", "torch", "cv2")
#: GPU utilisation (per cent) below which the card counts as idle at the start.
IDLE_UTILISATION_PCT = 10.0

StatReader = Callable[[], Dict[int, Tuple[int, int]]]
Smi = Callable[..., List[str]]


#: Model families with a with- and a without-flip (mirroring) setting; YOLO and CRAUNet have one setting.
FLIP_FAMILIES = frozenset({"segformer", "unet", "nnunet"})
CRAUNET = "craunet"
UNBATCHED = "_unbatched"


@dataclass(frozen=True)
class Pair:
    """One breast: its MLO and CC DICOM files (and, optionally, the reference ΔD of its landmarks)."""

    index: int
    study: str
    side: str
    mlo: str
    cc: str
    reference_delta_mm: float = float("nan")


@dataclass(frozen=True)
class Setting:
    """How a configuration runs its pipeline: test-time flip / mirroring, and one batched forward for both views."""

    tta: bool = False
    batched: bool = True


def pipeline_settings(name: str, family: str, batched: bool = True) -> Dict[str, Setting]:
    """Configuration names a pipeline offers -> their setting.

    Flip families: ``<name>_tta`` / ``<name>_no_tta``; YOLO and CRAUNet: ``<name>`` (one setting). Every name
    except CRAUNet's (one model per view: nothing to batch) also exists with the suffix ``_unbatched``: one
    forward per image and flip, to check what batching changes. ``batched`` is the setting of the plain names.
    """
    if family == CRAUNET:
        return {name: Setting(tta=False, batched=False)}
    base = ({f"{name}_tta": True, f"{name}_no_tta": False} if family in FLIP_FAMILIES else {name: False})
    out = {k: Setting(tta=v, batched=batched) for k, v in base.items()}
    out.update({f"{k}{UNBATCHED}": Setting(tta=v, batched=False) for k, v in base.items()})
    return out


class InvalidMeasurement(RuntimeError):
    """The session was not alone on its GPU; nothing it measured is valid."""

    def __init__(self, reason: str, detail: Any = None):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


# ── CPU cores ─────────────────────────────────────────────────────────────────────────────────────────────────

def read_proc_stat(path: str = "/proc/stat") -> Dict[int, Tuple[int, int]]:
    """Per logical CPU: (busy jiffies = all but idle and iowait, total jiffies)."""
    out: Dict[int, Tuple[int, int]] = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if p and p[0].startswith("cpu") and p[0] != "cpu":
                v = [int(x) for x in p[1:9]]
                out[int(p[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out


def parse_cpu_list(text: str) -> Tuple[int, ...]:
    """``"0,24"`` / ``"0-1"`` (sysfs list syntax) -> sorted CPU numbers."""
    out: List[int] = []
    for part in text.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
    return tuple(sorted(out))


def thread_siblings(cpu: int) -> Tuple[int, ...]:
    """Logical CPUs sharing ``cpu``'s physical core (``cpu`` alone when sysfs does not say)."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as fh:
            return parse_cpu_list(fh.read())
    except OSError:
        return (cpu,)


def select_cores(n: int, sampling_s: float, allowed: Optional[Iterable[int]] = None,
                 stat: StatReader = read_proc_stat, siblings: Callable[[int], Tuple[int, ...]] = thread_siblings,
                 sleep: Callable[[float], None] = time.sleep) -> Tuple[List[int], Dict[str, Any]]:
    """The ``n`` least-loaded physical cores over ``sampling_s`` seconds, one logical CPU of each.

    Cores are ranked by the summed busy fraction of their siblings, then the busiest sibling, then the CPU numbers;
    of each chosen core the least-busy allowed sibling is taken. Returns the sorted logical CPUs and a record of the
    choice.
    """
    if allowed is None:
        allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    allowed = sorted(allowed)
    a = stat()
    sleep(sampling_s)
    b = stat()
    load = {c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a if c in b}
    busy = lambda c: load.get(c, 0.0)  # noqa: E731
    groups = sorted({siblings(c) for c in allowed})
    ranked = sorted(groups, key=lambda g: (sum(busy(c) for c in g), max(busy(c) for c in g), g))
    chosen = sorted(min((c for c in g if c in allowed), key=lambda c: (busy(c), c)) for g in ranked[:n])
    info: Dict[str, Any] = {
        "busy_fraction_chosen": {str(c): round(busy(c), 3) for c in chosen},
        "busy_fraction_sibling": {str(c): (round(max(busy(k) for k in siblings(c) if k != c), 3)
                                           if len(siblings(c)) > 1 else None) for c in chosen},
        "busy_fraction_all_cpus_mean": round(sum(load.values()) / max(1, len(load)), 3),
        "sampling_s": sampling_s,
    }
    if len(chosen) < n:
        info["note"] = f"only {len(chosen)} physical cores available for {n} threads"
    return chosen, info


# ── machine state ─────────────────────────────────────────────────────────────────────────────────────────────

def run_nvidia_smi(*args: str) -> List[str]:
    """``nvidia-smi <args> --format=csv,noheader`` as lines (empty when the tool is missing)."""
    try:
        res = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [s for s in res.stdout.strip().splitlines() if s.strip()]


def load1() -> float:
    """1-minute load average (NaN where the platform has none)."""
    try:
        with open("/proc/loadavg") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        try:
            return float(os.getloadavg()[0])
        except (OSError, AttributeError):
            return float("nan")


def machine_snapshot(smi: Smi = run_nvidia_smi) -> Dict[str, Any]:
    """Load averages, wall-clock time and the GPUs with their compute processes (start / end of a session)."""
    try:
        with open("/proc/loadavg") as fh:
            avg = fh.read().split()[:3]
    except OSError:
        avg = [str(load1())]
    return {"loadavg": avg, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nvidia_smi": smi("--query-gpu=index,utilization.gpu,memory.used,memory.total"),
            "nvidia_smi_apps": smi("--query-compute-apps=gpu_uuid,pid,used_memory")}


# ── process set-up (before numpy / torch) ─────────────────────────────────────────────────────────────────────

def parse_device(device: str) -> Tuple[bool, Optional[str]]:
    """``"cpu"`` -> (False, None); ``"cuda"`` -> (True, None): the first visible GPU; ``"1"`` / ``"cuda:1"`` ->
    (True, "1"): physical GPU 1 (set as ``CUDA_VISIBLE_DEVICES``)."""
    d = str(device).strip().lower()
    if d == "cpu":
        return False, None
    if d in ("cuda", "gpu"):
        return True, None
    if d.startswith("cuda:"):
        d = d[5:]
    if d.isdigit():
        return True, d
    raise ValueError(f"device must be cpu, cuda, a GPU index or cuda:<index>, not {device!r}")


def choose_cores(threads: int, cores: Optional[Sequence[int]], sampling_s: float,
                 select: Callable[[int, float], Tuple[List[int], Dict[str, Any]]] = select_cores,
                 can_pin: Optional[bool] = None) -> Tuple[List[int], Dict[str, Any], int]:
    """``(logical CPUs to pin to, record, threads)``: threads always equal the number of pinned cores.

    Explicit ``cores`` give the thread count. Fewer free physical cores than ``threads`` lower the thread count
    (recorded). Without pinning support (not Linux) nothing is pinned and ``threads`` stays.
    """
    if can_pin is None:
        can_pin = hasattr(os, "sched_setaffinity")
    if not can_pin:
        return [], {"note": "this platform cannot pin a process to cores; not pinned"}, threads
    if cores:
        chosen = sorted({int(c) for c in cores})
        return chosen, {"explicit": True}, len(chosen)
    chosen, info = select(threads, sampling_s)
    if not chosen:
        return [], dict(info, note="no core found; not pinned"), threads
    if len(chosen) < threads:
        info.update(threads_requested=threads, threads_lowered_to=len(chosen))
    return chosen, info, len(chosen)


def prepare_process(device: str, threads: int, cores: Optional[Sequence[int]] = None, sampling_s: float = 5.0,
                    smi: Smi = run_nvidia_smi) -> Dict[str, Any]:
    """Pin this process and set its thread / device variables; call it before numpy, OpenCV or torch is imported.

    A GPU index is a physical index as nvidia-smi lists it: ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is set with
    ``CUDA_VISIBLE_DEVICES`` (unless the environment already chose an order). Returns the record stored in the
    session file (start snapshot, pinned CPUs and how they were chosen, the thread count actually used, whether a
    threaded library was already loaded, the device variables).
    """
    gpu, index = parse_device(device)
    preloaded = sorted(m for m in THREADED_MODULES if m in sys.modules)
    start = machine_snapshot(smi)
    chosen, info, threads = choose_cores(threads, cores, sampling_s)
    if chosen:
        os.sched_setaffinity(0, chosen)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = index
    for k in THREAD_ENV:
        os.environ[k] = str(threads)
    os.environ["YOLO_VERBOSE"] = "False"
    print("pinned to logical CPUs", chosen, info, f"threads {threads}", flush=True)
    if preloaded:
        print(f"warning: {preloaded} imported before pinning; their thread pools may run outside the pinned cores",
              flush=True)
    return {"start": start, "pinned_cpus": chosen, "core_choice": info, "threads": threads,
            "preloaded_before_pinning": preloaded, "gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_visible_devices_before": previous,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER") if gpu else None}


# ── jobs and order ────────────────────────────────────────────────────────────────────────────────────────────

def build_jobs(n_pairs: int, warmup: int, passes: int) -> List[Tuple[int, int, int]]:
    """``(job index, pass, pair index)``: pass 0 = the first ``warmup`` pairs, then passes 1..``passes`` over every
    pair (pass 1 over every pair, then again)."""
    jobs = [(0, c) for c in range(min(warmup, n_pairs))]
    for t in range(1, passes + 1):
        jobs += [(t, c) for c in range(n_pairs)]
    return [(j, t, c) for j, (t, c) in enumerate(jobs)]


def base_order(configurations: Sequence[str], seed: int) -> List[str]:
    """The session's fixed random order of the configurations."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [configurations[i] for i in rng.permutation(len(configurations))]


def rotated(base: Sequence[str], job_index: int) -> List[str]:
    """Order of job ``job_index``: ``base`` rotated left by ``job_index`` positions."""
    if not base:
        return []
    k = job_index % len(base)
    return list(base[k:]) + list(base[:k])


# ── GPU guard ─────────────────────────────────────────────────────────────────────────────────────────────────

class GpuGuard:
    """Watches the compute processes on one card (by UUID)."""

    def __init__(self, uuid: str, allowed_pids: Iterable[int] = (), smi: Smi = run_nvidia_smi,
                 own_pid: Optional[int] = None, uuid_from: str = "", visible: str = "", name: str = ""):
        self.uuid, self.smi = uuid, smi
        self.allowed = {int(p) for p in allowed_pids}
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self.info: Dict[str, Any] = {"uuid": uuid, "uuid_from": uuid_from, "visible": visible, "name": name,
                                     "allowed_pids": sorted(self.allowed), "checks": 0}

    @classmethod
    def for_visible_device(cls, torch_uuid: Optional[str], visible: Optional[str], allowed_pids: Iterable[int] = (),
                           smi: Smi = run_nvidia_smi, name: str = "", own_pid: Optional[int] = None) -> GpuGuard:
        """The card torch uses as ``cuda:0``: its UUID from torch when nvidia-smi knows it, else the first entry
        of ``CUDA_VISIBLE_DEVICES`` (an index or a UUID)."""
        known = {}
        for line in smi("--query-gpu=index,uuid"):
            idx, _, uid = line.partition(",")
            known[idx.strip()] = uid.strip()
        first = (visible or "").split(",")[0].strip()
        by_visible = first if first.startswith("GPU-") else known.get(first or "0")
        if torch_uuid and torch_uuid in known.values():
            if first and by_visible and by_visible != torch_uuid:
                raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} is {by_visible} for nvidia-smi but {torch_uuid} "
                                   "for CUDA: set CUDA_DEVICE_ORDER=PCI_BUS_ID so both count the cards alike")
            uuid, source = torch_uuid, "torch"
        elif by_visible:
            uuid, source = by_visible, "CUDA_VISIBLE_DEVICES"
        else:
            raise RuntimeError(f"cannot identify the GPU (CUDA_VISIBLE_DEVICES={visible!r}; nvidia-smi lists "
                               f"{sorted(known)})")
        return cls(uuid, allowed_pids, smi, own_pid, uuid_from=source, visible=visible or "", name=name)

    def processes(self) -> List[Dict[str, Any]]:
        """Compute processes on this card: ``{pid, used, name}``."""
        out = []
        for line in self.smi("--query-compute-apps=gpu_uuid,pid,used_memory,process_name"):
            q = [x.strip() for x in line.split(",")]
            if len(q) >= 3 and q[0] == self.uuid:
                out.append({"pid": int(q[1]), "used": q[2], "name": q[3] if len(q) > 3 else ""})
        return out

    def foreign(self) -> List[Dict[str, Any]]:
        """Processes on this card that are neither this process nor an allowed persistent one."""
        return [r for r in self.processes() if r["pid"] != self.own_pid and r["pid"] not in self.allowed]

    def usage(self) -> Dict[str, Any]:
        for line in self.smi("--query-gpu=uuid,utilization.gpu,memory.used"):
            q = [x.strip() for x in line.split(",")]
            if q and q[0] == self.uuid and len(q) >= 3:
                return {"utilization_pct": float(q[1].split()[0]), "memory_used": q[2]}
        return {}

    def check_start(self, polls: int = 30, poll_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> None:
        """Raise :class:`InvalidMeasurement` when a foreign process is on the card, or when the card stays busy
        (utilisation >= 10 %) for ``polls`` x ``poll_s`` seconds although no foreign process is listed."""
        self.info["apps_start"] = self.processes()
        found = self.foreign()
        if found:
            raise InvalidMeasurement("another compute process is on the card at start", found)
        for _ in range(polls):
            self.info["usage_start"] = self.usage()
            if self.info["usage_start"].get("utilization_pct", 0.0) < IDLE_UTILISATION_PCT:
                return
            sleep(poll_s)
        raise InvalidMeasurement("the card is busy at start although no foreign process is listed "
                                 "(a persistent process is computing)", self.info.get("usage_start"))

    def check(self) -> List[Dict[str, Any]]:
        """One periodic check: the foreign processes now on the card (empty = fine)."""
        self.info["checks"] += 1
        return self.foreign()

    def finish(self) -> List[Dict[str, Any]]:
        """End of the session: record the card state and return any foreign process."""
        self.info["apps_end"], self.info["usage_end"] = self.processes(), self.usage()
        return self.foreign()


def allowed_pid_list(text: Optional[str]) -> List[int]:
    """``"12,34"`` -> [12, 34]."""
    return [int(x) for x in str(text or "").split(",") if x.strip()]


def mapping_arg(items: Optional[Sequence[str]], what: str) -> Mapping[str, str]:
    """``["a=x", "b=y"]`` -> {"a": "x", "b": "y"} (repeatable ``NAME=VALUE`` flags)."""
    out: Dict[str, str] = {}
    for item in items or ():
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"{what}: expected NAME=VALUE, got {item!r}")
        out[key.strip()] = value.strip()
    return out
