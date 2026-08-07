"""基于 pandapower 的电力层仿真器 (IEEE 33 或任意 pandapower 网络).

实现 interfaces.PowerSimulator 契约, 引擎与下游指标代码零改动接入.

建模选择 (与草稿"基于动态流与容量约束的方法"对应):
  * 准稳态级联: 每个时间步在 step() 内部求解
        损伤状态 → 供电性检查 → 潮流 → 过载跳闸 → 再潮流
    的平衡态 (Motter-Lai 式), 引擎看不到中间态;
  * 无记忆平衡: 每步从"完好拓扑 − 当前损伤"重新求平衡, 跳闸不跨步
    残留 —— F_j(t) 因此是当前损伤状态的确定函数, M/T 可解释、线性
    验证干净. 若需"断路器不自动重合"的持久跳闸, 在 _solve_equilibrium
    里保留 tripped 集合即可, 接口不变;
  * 线路容量 (Motter-Lai 变体):
        C_l = (1+alpha) · I_l(0) + beta · mean(I(0))
    alpha 为容忍系数; beta·mean 是容量下限项, 避免基态零流线路
    (如闭合运行的联络线) 容量为零、一扰动即跳闸的退化情形;
  * 节点失效语义:
        d >= D_FULL  → 母线整体退出运行 (完全失效);
        0 < d < D_FULL → 相邻线路容量削减 ×(1−d) (部分失效),
                         可诱发过载级联 —— 这是线性验证的对象;
  * 节点功能 F_j = 带电指示 × (1 − 自身损伤);
    充电供给按电压水平在 [vmin_zero, vmin_full] 间线性折减.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

try:  # pandapower 版本兼容
    from pandapower import LoadflowNotConverged
except ImportError:  # pragma: no cover
    from pandapower.powerflow import LoadflowNotConverged


PANDAPOWER_CASE_ALIASES = {
    "ieee33": ("case33bw",),
    "ieee33bw": ("case33bw",),
    "case33": ("case33bw",),
    "case33bw": ("case33bw",),
    "ieee123": ("case_ieee123", "case123", "ieee123", "IEEE123"),
    "case123": ("case_ieee123", "case123", "ieee123", "IEEE123"),
}


def load_pandapower_network(case_name: str = "case33bw", path: str | Path | None = None):
    """Load a pandapower network from a built-in case name or a local file."""
    if path is not None:
        net_path = Path(path)
        suffix = net_path.suffix.lower()
        if suffix == ".json":
            return pp.from_json(str(net_path))
        if suffix in {".p", ".pickle", ".pkl"}:
            return pp.from_pickle(str(net_path))
        raise ValueError(f"unsupported pandapower network file type: {net_path}")

    normalized = case_name.lower()
    candidates = PANDAPOWER_CASE_ALIASES.get(normalized, (case_name,))
    tried = []
    for candidate in candidates:
        tried.append(candidate)
        factory = getattr(pn, candidate, None)
        if callable(factory):
            return factory()
    available = sorted(name for name in dir(pn) if name.startswith("case") or "ieee" in name.lower())
    raise ValueError(
        f"pandapower network case '{case_name}' is not available in this environment. "
        f"Tried: {tried}. Pass path=... with a pandapower .json/.p/.pickle network, "
        f"or install a pandapower version that provides the case. Available examples: {available[:20]}"
    )


def load_ieee123_pandapower_network(path: str | Path | None = None):
    """Load IEEE123 as a pandapower net when available."""
    return load_pandapower_network("ieee123", path=path)


class PandapowerPowerSim:
    layer = "E"
    D_FULL = 0.999  # 损伤达到该值视为完全失效(母线退出运行)

    def __init__(
        self,
        net=None,
        station_rated: Mapping[str, float] | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        close_tie_lines: bool = True,
        vmin_zero: float = 0.80,
        vmin_full: float = 0.90,
        max_cascade_iter: int = 25,
    ):
        self.net = net if net is not None else load_pandapower_network("case33bw")
        self.close_tie_lines = bool(close_tie_lines)
        if self.close_tie_lines:
            # 联络线闭合 → 网状运行, 故障后存在替代路径, 级联机制才有意义;
            # False 则为纯辐射状运行, 故障 = 下游整体失电.
            self.net.line.loc[:, "in_service"] = True

        self.station_rated = {str(k): float(v) for k, v in (station_rated or {}).items()}
        self.alpha, self.beta = float(alpha), float(beta)
        self.vmin_zero, self.vmin_full = float(vmin_zero), float(vmin_full)
        self.max_cascade_iter = int(max_cascade_iter)

        # 完好拓扑快照 (每次求平衡都从这里出发)
        self._base_bus_is = self.net.bus.in_service.copy()
        self._base_line_is = self.net.line.in_service.copy()

        # 基态潮流 → 线路容量
        self._runpp()
        base_i = self.net.res_line.i_ka.fillna(0.0).to_numpy()
        self.capacity_ka = (1.0 + self.alpha) * base_i + self.beta * float(base_i.mean())

        self.damage: dict = {}
        self._energized = None
        self._vm = None
        self._sig = None  # 上次求平衡时的损伤签名 (状态未变则跳过重解)
        self.reset()

    # ══════════════════════ LayerSimulator 契约 ══════════════════════
    def reset(self, seed: int = 0) -> None:
        self.damage = {str(b): 0.0 for b in self.net.bus.index}
        self._sig = None
        self._solve_equilibrium()

    def node_ids(self):
        return [str(b) for b in self.net.bus.index]

    def apply_shock(self, node_id: str, severity: float) -> None:
        node_id = str(node_id)
        self.damage[node_id] = min(1.0, max(self.damage[node_id], float(severity)))

    def step(self, t: float, dt: float) -> list:
        """准稳态推进: 按当前损伤状态求级联平衡, 返回本步事件."""
        return self._solve_equilibrium()

    def node_functions(self) -> Mapping[str, float]:
        F = {}
        for b in self.net.bus.index:
            sb = str(b)
            on = 1.0 if bool(self._energized.at[b]) else 0.0
            F[sb] = on * (1.0 - min(self.damage[sb], 1.0))
        return F

    def damage_levels(self) -> Mapping[str, float]:
        return dict(self.damage)

    def repair(self, node_id: str, amount: float) -> float:
        node_id = str(node_id)
        used = min(float(amount), self.damage[node_id])
        self.damage[node_id] -= used
        return used

    # ══════════════════════ PowerSimulator 契约 ══════════════════════
    def charging_supply(self) -> Mapping[str, float]:
        """充电站母线 → 当前可用供电功率 (kW), 按电压水平折减."""
        sup = {}
        for sb, rated in self.station_rated.items():
            b = int(sb)
            if not bool(self._energized.at[b]):
                sup[sb] = 0.0
                continue
            vm = float(self._vm.at[b]) if np.isfinite(self._vm.at[b]) else 0.0
            vf = np.clip(
                (vm - self.vmin_zero) / max(self.vmin_full - self.vmin_zero, 1e-9),
                0.0,
                1.0,
            )
            sup[sb] = rated * float(vf) * (1.0 - min(self.damage[sb], 1.0))
        return sup

    # ═══════════════════════════ 内部求解 ═══════════════════════════
    def _table_rows(self, name: str, cols: list[str]) -> list:
        tbl = getattr(self.net, name, None)
        if tbl is None:
            return []
        present = [c for c in cols if c in tbl.columns]
        rows = []
        for idx, row in tbl[present].sort_index().iterrows():
            vals = {c: row[c] for c in present}
            rows.append({"index": str(idx), **vals})
        return rows

    def _net_digest(self) -> str:
        payload = {
            "bus_index": [str(i) for i in self.net.bus.index],
            "bus_base_in_service": {
                str(i): bool(v) for i, v in self._base_bus_is.sort_index().items()
            },
            "line_base_in_service": {
                str(i): bool(v) for i, v in self._base_line_is.sort_index().items()
            },
            "bus": self._table_rows("bus", ["vn_kv", "type", "zone"]),
            "line": self._table_rows(
                "line",
                [
                    "from_bus", "to_bus", "length_km", "r_ohm_per_km",
                    "x_ohm_per_km", "c_nf_per_km", "max_i_ka",
                ],
            ),
            "load": self._table_rows(
                "load", ["bus", "p_mw", "q_mvar", "scaling", "in_service"]
            ),
            "ext_grid": self._table_rows(
                "ext_grid", ["bus", "vm_pu", "in_service"]
            ),
            "trafo": self._table_rows(
                "trafo", ["hv_bus", "lv_bus", "sn_mva", "in_service"]
            ),
            "sgen": self._table_rows(
                "sgen", ["bus", "p_mw", "q_mvar", "scaling", "in_service"]
            ),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def signature(self) -> dict:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "net_digest": self._net_digest(),
            "station_rated": sorted(self.station_rated.items()),
            "alpha": self.alpha,
            "beta": self.beta,
            "close_tie_lines": self.close_tie_lines,
            "vmin_zero": self.vmin_zero,
            "vmin_full": self.vmin_full,
            "max_cascade_iter": self.max_cascade_iter,
            "D_FULL": self.D_FULL,
        }

    def _runpp(self) -> bool:
        try:
            pp.runpp(self.net, numba=False, init="auto")
            return True
        except LoadflowNotConverged:
            return False

    def _blackout(self, events: list, reason: str) -> list:
        """保守回退: 本步全网视为失电 (潮流不收敛等病态情形)."""
        self._energized = self.net.bus.in_service.copy()
        self._energized.loc[:] = False
        import pandas as pd

        self._vm = pd.Series(0.0, index=self.net.bus.index)
        events.append({"type": "blackout_fallback", "reason": reason})
        return events

    def _solve_equilibrium(self) -> list:
        # 损伤状态未变则跳过重解 (warm-up 与冲击前的空转步共用缓存)
        sig = tuple(sorted(self.damage.items()))
        if sig == self._sig:
            return []
        self._sig = sig

        net, events = self.net, []

        # 1) 从完好拓扑出发, 施加当前损伤
        net.bus["in_service"] = self._base_bus_is.copy()
        net.line["in_service"] = self._base_line_is.copy()
        full_out = [b for b in net.bus.index if self.damage[str(b)] >= self.D_FULL]
        if full_out:
            net.bus.loc[full_out, "in_service"] = False

        # 部分失效 → 相邻线路容量削减 (取两端较严重者)
        cap = self.capacity_ka.copy()
        for pos, li in enumerate(net.line.index):
            df = self.damage[str(net.line.at[li, "from_bus"])]
            dt_ = self.damage[str(net.line.at[li, "to_bus"])]
            cap[pos] *= min(1.0 - df, 1.0 - dt_)

        # 平衡台架不存在 (变电站/松弛母线被毁) → 全网失电
        eg = net.ext_grid[net.ext_grid.in_service]
        if not any(bool(net.bus.at[b, "in_service"]) for b in eg.bus):
            return self._blackout(events, "slack_bus_out")

        # 2) 级联循环: 供电性检查 → 潮流 → 过载跳闸, 直至无新跳闸
        stable = False
        for cascade_iter in range(self.max_cascade_iter):
            unsup = top.unsupplied_buses(net)
            if unsup:
                net.bus.loc[sorted(unsup), "in_service"] = False
            if not self._runpp():
                return self._blackout(events, "pf_diverged")

            i_ka = net.res_line.i_ka.fillna(0.0)
            in_srv = net.line.in_service
            over = [
                li
                for pos, li in enumerate(net.line.index)
                if bool(in_srv.at[li]) and float(i_ka.at[li]) > cap[pos] + 1e-9
            ]
            if not over:
                stable = True
                break
            net.line.loc[over, "in_service"] = False
            for li in over:
                pos = list(net.line.index).index(li)
                events.append(
                    {
                        "type": "line_trip",
                        "target": f"E:line{li}",
                        "from_bus": int(net.line.at[li, "from_bus"]),
                        "to_bus": int(net.line.at[li, "to_bus"]),
                        "i_ka": round(float(i_ka.at[li]), 4),
                        "capacity_ka": round(float(cap[pos]), 4),
                        "cascade_iter": cascade_iter,
                    }
                )

        if not stable:
            unsup = top.unsupplied_buses(net)
            if unsup:
                net.bus.loc[sorted(unsup), "in_service"] = False
            if not self._runpp():
                return self._blackout(events, "pf_diverged_after_max_iter")
            i_ka = net.res_line.i_ka.fillna(0.0)
            in_srv = net.line.in_service
            over = [
                li
                for pos, li in enumerate(net.line.index)
                if bool(in_srv.at[li]) and float(i_ka.at[li]) > cap[pos] + 1e-9
            ]
            if over:
                events.append(
                    {
                        "type": "cascade_not_converged",
                        "remaining_overloads": len(over),
                        "max_iter": self.max_cascade_iter,
                    }
                )
                return self._blackout(events, "cascade_not_converged")

        # 3) 缓存平衡态 (供 node_functions / charging_supply 查询)
        self._energized = net.bus.in_service.copy()
        self._vm = net.res_bus.vm_pu.copy()
        return events

