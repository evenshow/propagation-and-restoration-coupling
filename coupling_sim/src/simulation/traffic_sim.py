"""基于用户均衡 (UE) 的交通层仿真器 —— Sioux Falls 或任意 TNTP 网络.

实现 interfaces.TrafficSimulator 契约, 引擎与下游指标代码零改动接入.

建模选择 (与草稿"电网跑潮流, 交通跑用户均衡"的方案对应):
  * 准稳态: 每个时间步用 Frank-Wolfe 求解 BPR 用户均衡, 得到
    链路流量与走行时间 —— 对应电力侧的"潮流";
  * 跨层函数 (电 → 交通): 充电站服务能力
        cap_veh_now = cap_veh × clip(S_j / rated_kw, 0, 1) × (1 − d_j)
    本地需求超出部分 U_j 为未满足充电需求; U_j 以额外 OD 出行的
    形式改道去最近的存活备用站 (绕行 → 拥堵), 备用站全灭则整体
    弃行 (只计入服务损失, 不加流量);
  * 节点功能:
        充电站节点  F_j = 本地被服务比例 served_j / demand_j
        普通道路节点 F_j = 相邻链路 t0/t 的流量加权平均 (平均速度比代理)
    引擎会用 warm-up 稳态 F_j0 归一化, 所以基态拥堵不影响 p 的起点;
  * 交通节点受冲击 (shock_layer="T"): 相邻链路容量 ×(1−d) (道路封闭);
  * 一阶惯性 (时间常数 tau): 拥堵的积累与消散需要时间, 使
    T_i→j 的损失时间质心有非平凡取值;
  * UE 求解按 (充电供给, 损伤) 签名缓存, 状态未变不重解.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..networks.traffic_layer import TrafficNetwork, load_tntp_network


# ══════════════════════ Frank-Wolfe 用户均衡 ══════════════════════
class UserEquilibrium:
    """经典 Frank-Wolfe: 全有全无加载 + 精确线搜索, BPR 走行时间."""

    def __init__(self, net: TrafficNetwork, max_iter: int = 60, gap_tol: float = 2e-3):
        self.net = net
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self._link_of = {
            (int(f), int(t)): k for k, (f, t) in enumerate(zip(net.fb, net.tb))
        }

    def bpr(self, x: np.ndarray, cap_scale: np.ndarray) -> np.ndarray:
        c = np.maximum(self.net.cap * cap_scale, 1e-3)
        return self.net.t0 * (1.0 + self.net.bpr_b * (x / c) ** self.net.bpr_p)

    def _aon(self, costs: np.ndarray, od: np.ndarray) -> np.ndarray:
        """全有全无加载: 按当前链路成本的最短路装载全部 OD."""
        n = self.net.n_nodes
        g = csr_matrix((costs, (self.net.fb, self.net.tb)), shape=(n, n))
        origins = np.where(od.sum(axis=1) > 0)[0]
        _, pred = dijkstra(
            g, directed=True, indices=origins, return_predecessors=True
        )
        x = np.zeros(self.net.n_links)
        for oi, o in enumerate(origins):
            po = pred[oi]
            for d in np.where(od[o] > 0)[0]:
                if d == o or po[d] < 0:
                    continue
                dem, v = od[o, d], int(d)
                while v != o:
                    u = int(po[v])
                    x[self._link_of[(u, v)]] += dem
                    v = u
        return x

    def solve(self, od: np.ndarray, cap_scale: np.ndarray | None = None):
        """返回 (链路流量 x, 链路走行时间 t)."""
        cs = np.ones(self.net.n_links) if cap_scale is None else cap_scale
        x = self._aon(self.net.t0, od)
        for _ in range(self.max_iter):
            t = self.bpr(x, cs)
            y = self._aon(t, od)
            # 相对间隙
            xt, yt = float(x @ t), float(y @ t)
            if xt > 0 and (xt - yt) / xt < self.gap_tol:
                break
            # 精确线搜索: 在 d=y-x 方向上求 dZ/dλ=Σ d·t(x+λd)=0
            d = y - x
            lo, hi = 0.0, 1.0
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if float(d @ self.bpr(x + mid * d, cs)) > 0:
                    hi = mid
                else:
                    lo = mid
            x = x + 0.5 * (lo + hi) * d
        return x, self.bpr(x, cs)


# ══════════════════════ 交通层仿真器 ══════════════════════
class TNTPTrafficSim:
    layer = "T"

    def __init__(
        self,
        data_dir,
        stations: Mapping[str, Mapping] | None = None,
        prefix: str | None = None,
        network_name: str | None = None,
        tau: float = 15.0,
        divert_alive_threshold: float = 0.05,
        sig_round: float = 0.02,
        min_capacity_floor: float = 0.0,
    ):
        """stations: {交通节点id: dict(rated_kw, demand_veh, cap_veh)}.

        rated_kw   与跨层接口/电力侧一致的额定功率 (换算服务比例用);
        demand_veh 该站的 EV 充电需求 (veh/h);
        cap_veh    满功率时的服务能力 (veh/h).
        备用站 = 自由流最短时间下最近的另一站 (reset 时计算).
        """
        self.net: TrafficNetwork = load_tntp_network(data_dir, prefix=prefix, network_name=network_name)
        self.ue = UserEquilibrium(self.net)
        self.stations = {str(k): dict(v) for k, v in (stations or {}).items()}
        self.tau = float(tau)
        self.divert_alive_threshold = float(divert_alive_threshold)
        self.sig_round = float(sig_round)
        self.min_capacity_floor = float(min_capacity_floor)
        if not 0.0 <= self.min_capacity_floor <= 1.0:
            raise ValueError("min_capacity_floor must be in [0, 1]")

        self._backup = self._nearest_backups()
        self.F: dict = {}
        self.damage: dict = {}
        self.S: dict = {}
        self._sig = None
        self._target: dict = {}
        self._road_target: dict = {}
        self.reset()

    def _net_digest(self) -> str:
        payload = {
            "node_ids": list(self.net.node_ids),
            "fb": self.net.fb.tolist(),
            "tb": self.net.tb.tolist(),
            "cap": self.net.cap.tolist(),
            "t0": self.net.t0.tolist(),
            "bpr_b": self.net.bpr_b.tolist(),
            "bpr_p": self.net.bpr_p.tolist(),
            "od": self.net.od.tolist(),
            "coords": {k: list(v) for k, v in sorted(self.net.coords.items())},
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def signature(self) -> dict:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "network_name": self.net.name,
            "net_digest": self._net_digest(),
            "stations": {k: dict(sorted(v.items())) for k, v in sorted(self.stations.items())},
            "tau": self.tau,
            "divert_alive_threshold": self.divert_alive_threshold,
            "sig_round": self.sig_round,
            "min_capacity_floor": self.min_capacity_floor,
            "ue_max_iter": self.ue.max_iter,
            "ue_gap_tol": self.ue.gap_tol,
        }

    # ---------------------------------------------------------- 内部
    def _nearest_backups(self) -> dict:
        """每个充电站按自由流最短时间找最近的另一站作为备用."""
        if len(self.stations) < 2:
            return {k: None for k in self.stations}
        n = self.net.n_nodes
        g = csr_matrix((self.net.t0, (self.net.fb, self.net.tb)), shape=(n, n))
        dist = dijkstra(g, directed=True)
        out = {}
        for j in self.stations:
            pj = self.net.pos(j)
            best, best_d = None, np.inf
            for k in self.stations:
                if k == j:
                    continue
                dd = dist[pj, self.net.pos(k)]
                if dd < best_d:
                    best, best_d = k, dd
            out[j] = best
        return out

    def _station_fraction(self, j: str) -> float:
        cfg = self.stations[j]
        frac = np.clip(self.S.get(j, cfg["rated_kw"]) / cfg["rated_kw"], 0.0, 1.0)
        return float(frac) * (1.0 - min(self.damage[j], 1.0))

    def _solve_targets(self) -> None:
        """按当前 (充电供给, 损伤) 求 UE, 更新目标功能值."""
        # 1) 充电站服务与改道需求
        od = self.net.od.copy()
        station_target = {}
        for j, cfg in self.stations.items():
            cap_now = cfg["cap_veh"] * self._station_fraction(j)
            served = min(cfg["demand_veh"], cap_now)
            unserved = cfg["demand_veh"] - served
            station_target[j] = served / max(cfg["demand_veh"], 1e-9)
            bk = self._backup.get(j)
            if (
                unserved > 1e-9
                and bk is not None
                and self._station_fraction(bk) > self.divert_alive_threshold
            ):
                od[self.net.pos(j), self.net.pos(bk)] += unserved  # 绕行出行

        # 2) 道路封闭: 相邻链路容量削减
        dam = np.array([self.damage[nid] for nid in self.net.node_ids])
        cap_scale = np.minimum(1.0 - dam[self.net.fb], 1.0 - dam[self.net.tb])
        cap_scale = np.maximum(cap_scale, self.min_capacity_floor)

        # 3) 用户均衡
        x, t = self.ue.solve(od, cap_scale)

        # 4) 目标功能: 道路节点 = 相邻链路 t0/t 的流量加权平均
        ratio = self.net.t0 / np.maximum(t, 1e-9)
        target = {}
        for k, nid in enumerate(self.net.node_ids):
            inc = (self.net.fb == k) | (self.net.tb == k)
            w = x[inc]
            r = ratio[inc]
            f = float((w @ r) / w.sum()) if w.sum() > 1e-9 else float(r.mean())
            target[nid] = f * (1.0 - min(self.damage[nid], 1.0))
        self._road_target = dict(target)
        for j, sf in station_target.items():
            target[j] = sf  # 充电站节点以充电服务水平为功能
        self._target = target

    # ══════════════════ TrafficSimulator 契约 ══════════════════
    def reset(self, seed: int = 0) -> None:
        self.damage = {nid: 0.0 for nid in self.net.node_ids}
        self.S = {j: cfg["rated_kw"] for j, cfg in self.stations.items()}
        self._sig = None
        self._solve_targets()
        self.F = dict(self._target)  # 从基态均衡出发

    def node_ids(self):
        return list(self.net.node_ids)

    def apply_shock(self, node_id: str, severity: float) -> None:
        node_id = str(node_id)
        self.damage[node_id] = min(1.0, max(self.damage[node_id], float(severity)))

    def step(self, t: float, dt: float, charging_capacity: Mapping[str, float]) -> list:
        self.S.update(charging_capacity or {})
        # 签名缓存: 供给按 sig_round 粒度取整, 状态未变不重解 UE
        sig = (
            tuple(
                round(self._station_fraction(j) / self.sig_round)
                for j in sorted(self.stations)
            ),
            tuple(sorted(self.damage.items())),
        )
        if sig != self._sig:
            self._sig = sig
            self._solve_targets()
        a = dt / (self.tau + dt)  # 一阶惯性: 拥堵积累/消散需要时间
        for nid in self.net.node_ids:
            self.F[nid] += a * (self._target[nid] - self.F[nid])
        return []

    def node_functions(self) -> Mapping[str, float]:
        return dict(self.F)

    def damage_levels(self) -> Mapping[str, float]:
        return dict(self.damage)

    def repair(self, node_id: str, amount: float) -> float:
        node_id = str(node_id)
        used = min(float(amount), self.damage[node_id])
        self.damage[node_id] -= used
        return used


class SiouxFallsTrafficSim(TNTPTrafficSim):
    """Compatibility wrapper for the existing Sioux Falls workflows."""

    def __init__(
        self,
        data_dir,
        stations: Mapping[str, Mapping] | None = None,
        tau: float = 15.0,
        divert_alive_threshold: float = 0.05,
        sig_round: float = 0.02,
        min_capacity_floor: float = 0.0,
    ):
        super().__init__(
            data_dir,
            stations=stations,
            prefix="SiouxFalls",
            network_name="Sioux Falls",
            tau=tau,
            divert_alive_threshold=divert_alive_threshold,
            sig_round=sig_round,
            min_capacity_floor=min_capacity_floor,
        )

