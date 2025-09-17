
def compute_footprint(runtime_sec: float, peak_kb: float):
    PUE = 1.67
    PSF = 1
    n_CPUcores = 8
    CPUpower = 15.6
    usageCPU_used = 1
    memoryPower = 0.3725
    carbonIntensity = 415.6  # gCO2/kWh

    mem_gb = peak_kb / (1024.0 * 1024.0)

    power_core = PUE * n_CPUcores * CPUpower * usageCPU_used
    power_mem  = PUE * (mem_gb * memoryPower)
    power_tot  = power_core + power_mem

    energy_core = runtime_sec * power_core * PSF / 1000.0
    energy_mem  = runtime_sec * power_mem  * PSF / 1000.0
    energy_tot  = runtime_sec * power_tot  * PSF / 1000.0

    ce_core = energy_core * carbonIntensity
    ce_mem  = energy_mem  * carbonIntensity
    ce_tot  = energy_tot  * carbonIntensity

    r = lambda x, n=4: round(x, n)
    energy_tot = r(energy_tot); ce_tot = r(ce_tot)
    ce_core = r(ce_core); ce_mem = r(ce_mem)

    if (ce_core + ce_mem) > 0:
        ce_core_per = round(ce_core / (ce_core + ce_mem) * 100.0, 2)
        ce_mem_per  = round(ce_mem  / (ce_core + ce_mem) * 100.0, 2)
    else:
        ce_core_per = ce_mem_per = 0.0

    tree_days   = r(ce_tot * 30 / 11000 * 12)
    train_km    = r(ce_tot / 41)
    lightbulb_h = r(energy_tot * 1000 / 60)
    car_km      = r(ce_tot / 251)

    return {
        "energy_kwh": energy_tot,
        "carbon_g": ce_tot,
        "ce_core_g": ce_core,
        "ce_mem_g": ce_mem,
        "ce_core_per": ce_core_per,
        "ce_mem_per": ce_mem_per,
        "tree_days": tree_days,
        "train_km": train_km,
        "lightbulb_h": lightbulb_h,
        "car_km": car_km,
    }

def _pct_reduction(old: float | None, new: float | None) -> float:
    if old is None or new is None:
        return 0.0
    if old <= 0:
        return 0.0
    return (old - new) / old * 100.0

def _pos(x: float | None) -> float:
    if x is None:
        return 0.0
    return x if x > 0 else 0.0

def compute_savings(base_rt, base_mem_kb, base_fp: dict, opt_rt, opt_mem_kb, opt_fp: dict):
    saved = {
        "runtime_s": (base_rt - opt_rt) if (base_rt is not None and opt_rt is not None) else None,
        "memory_kb": (base_mem_kb - opt_mem_kb) if (base_mem_kb is not None and opt_mem_kb is not None) else None,
        "energy_kwh": base_fp["energy_kwh"] - opt_fp["energy_kwh"],
        "carbon_g":   base_fp["carbon_g"]   - opt_fp["carbon_g"],
        "tree_days":  base_fp["tree_days"]  - opt_fp["tree_days"],
        "train_km":   base_fp["train_km"]   - opt_fp["train_km"],
        "lightbulb_h":base_fp["lightbulb_h"]- opt_fp["lightbulb_h"],
        "car_km":     base_fp["car_km"]     - opt_fp["car_km"],
    }
    pct = {
        "runtime": _pct_reduction(base_rt, opt_rt),
        "memory":  _pct_reduction(base_mem_kb, opt_mem_kb),
        "energy":  _pct_reduction(base_fp["energy_kwh"], opt_fp["energy_kwh"]),
        "carbon":  _pct_reduction(base_fp["carbon_g"],   opt_fp["carbon_g"]),
    }
    return saved, pct
