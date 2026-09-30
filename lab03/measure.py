from __future__ import annotations

import statistics
import math
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()

    elps_times = []
    for _ in range(repeats):
        start_time = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end_time = bench.clock()

        elps_times.append((end_time-start_time)/1000000.0)

    return elps_times


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("too few samples")
    
    snd_hlf = samples[len(samples)//2:]
    med = statistics.median(snd_hlf)

    if med<= 0:
        return unknown("median is less than 0")
    
    threshold = med * (1+ WARMUP_TOL)
    count = 0
    consec = True
    for i in samples:
        if i > threshold:
            count += 1
    return {
        "value": 1,
        "source": "leading prefix above (1 + 0.5) x median of the run's second half",
        "status": "ok",
        "settled_rate_ms": med,
        "threshold_ms": threshold,
        "tolerance": WARMUP_TOL,
        "retained": count
    }

    pass

def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return{
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None
        }
    
    samples = sorted(samples)
    mean = statistics.fmean(samples)
    min_val = min(samples)
    max_val = max(samples)

    std_dev = statistics.stdev(samples) if len(samples) > 2 else 0.0

    h_50 = (len(samples) - 1) *50/100
    i_50 = math.floor(h_50)
    val_50 = samples[i_50] + (h_50 - i_50) *(samples[i_50 + 1] - samples[i_50])

    h_95 = (len(samples) - 1) * 95/100
    i_95 = math.floor(h_95)
    val_95 = samples[i_95] + (h_95 - i_95) *(samples[i_95 + 1] - samples[i_95])

    h_99 = (len(samples) - 1) *99/100
    i_99 = math.floor(h_99)
    val_99 = samples[i_99] + (h_99 - i_99) *(samples[i_99 + 1] - samples[i_99])
    return{
            "n": len(samples),
            "mean" : mean,
            "std": std_dev,
            "min": min_val,
            "max": max_val,
            "p50": round(val_50, 4),
            "p95": round(val_95, 4),
            "p99": round(val_99, 4)
        }
    
    

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 20:
        return unknown("there are not enough samples")
    
    samples = sorted(samples)

    trim = int(len(samples) * 0.05)

    trimmed = samples[trim: len(samples)-trim]
    gaps = []
    for i in range(len(trimmed) - 2):
        gaps.append(trimmed[i+1] - trimmed[i])
    
    med_gap = statistics.median(gaps)
    if med_gap <=0 :
        return unknown("the timer resolution is too coarse")
    
    widst_gap = max(gaps)

    ratio = widst_gap/med_gap
    split_pt = gaps.index(widst_gap)
    left = split_pt + 1
    right = len(samples) - left

    return {
        "value": True if ratio >= 20.0 and left >= 0.1*len(samples) and right>=0.1*len(samples) else False,
        "source": "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        "status": "ok",
        "gap_ratio": ratio,
        "widest_gap_ms": widst_gap,
        "typical_gap_ms": med_gap,
        "modes": [
            {
                "n": left,
                "share": left/len(samples),
                "median_ms": statistics.median(samples[:left])
            },
            {
                "n": right,
                "share": right/len(samples),
                "median_ms": statistics.median(samples[left:])
            }
        ]
    }

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    out = bench.runner("nvpmodel", "-q")
    if out.returncode != 0:
        return unknown("missing sudo permissions", "end")
    lines = out.splitlines()
    mode_id = 0
    power_mode = None
    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            power_mode = line.split(":", 1)[1].strip()
            mode_id = int(lines[i + 1].strip())
            break
    min_freq = read_text(CPUFREQ_MIN)
    max_freq = read_text(CPUFREQ_MAX)

    return {
        "value": power_mode,
        "source": "nvpmodel -q",
        "status": "ok",
        "mode_index": mode_id,
        "jetson_clocks": True if min_freq == max_freq else False,
        "jetson_clocks_source": {
            "value": f"scaling_min_freq={min_freq}, scaling_max_freq={max_freq}",
            "source": "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq vs sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
            "status": "ok"
        }
    }



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    """Read temperature, board power consumption, and GPU utilization."""

    thermal_root = bench.telemetry / "sys/devices/virtual/thermal"

    temperatures = []

    if thermal_root.exists():
        for temp_file in thermal_root.glob("thermal_zone*/temp"):
            try:
                with open(temp_file, "r") as f:
                    raw_temp = f.read().strip()

                temp = int(raw_temp) / 1000.0

                # Ignore disabled sensors
                if temp <= -1000:
                    continue

                temperatures.append(temp)

            except (OSError, ValueError):
                continue

    if temperatures:
        highest_temp = max(temperatures)

        temperature = {
            "value": highest_temp,
            "source": thermal_root,
            "status": "ok"
        }
    else:
        temperature = unknown(
            "sys/devices/virtual/thermal",
            "no valid thermal zones found"
        )

    power_raw = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)

    if power_raw is None:
        power = unknown(
            "INA3221",
            "no power sensor file found"
        )
    else:
        try:
            power_mw = int(power_raw)

            power = {
                "value": power_mw,
                "source": "INA3221",
                "status": "ok"
            }

        except ValueError:
            power = unknown(
                "INA3221",
                "power sensor value was not an integer"
            )

    gpu_source, gpu_raw = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)

    if gpu_raw is None:
        gpu = unknown(
            "GPU load",
            "no GPU load file found"
        )
    else:
        try:
            gpu_percent = int(gpu_raw) / 10.0

            gpu = {
                "value": gpu_percent,
                "source": "GPU load",
                "status": "ok"
            }

        except ValueError:
            gpu = unknown(
                "GPU load",
                "GPU load value was not an integer"
            )

    return {
        # "temperature": temperature,
        "power": power,
        "gpu": gpu
    }

## for debugging - uncomment the following lines for debugging.
if __name__ == "__main__":
    env = Bench.real()
    # samples = run_timed_iterations(env, repeats=100)

    out = probe_telemetry(env)

    print(out)

# for generating system_report.json
# if __name__ == "__main__":
#     # calling base environment
#     env = Bench.real()

#     # get your samples
#     samples = run_timed_iterations(env, repeats=100)

#     # testing measurments and probes
#     report = {
#         "warmup_boundary": find_warmup_boundary(samples),
#         "summarize_setup": summarize(samples),
#         "is_multimodal": is_multimodal(samples),
#         "probe_power_state": probe_power_state(env),
#         "probe_telemetry": probe_telemetry(env),
#     }

#     # save samples
#     path = "samples_analysis.json"
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(samples, f, indent=4)

#     # save report
#     path = "system_report.json"
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(report, f, indent=4)