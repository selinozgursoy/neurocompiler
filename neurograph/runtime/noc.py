"""A packet-switched 2D mesh with dimension-order (X then Y) routing.

Traffic within a simulation step is delivered by a small discrete-event
simulation: each link carries one flit per hop time, so bursts that share a
link queue behind each other. Dimension-order routing keeps the channel
dependency graph acyclic, which is why the mesh cannot deadlock on routing.
"""
from __future__ import annotations

import heapq
from collections import defaultdict


class MeshNoC:
    def __init__(self, width: int, height: int, t_hop_ns: float, flit_bytes: int = 4):
        self.w, self.h, self.t_hop, self.flit_bytes = width, height, t_hop_ns, flit_bytes
        self.link_flits: dict[tuple, int] = defaultdict(int)   # lifetime load per directed link
        self.total_flit_hops = 0
        self.total_bytes = 0.0
        self._bursts: list[tuple[float, tuple, tuple, int]] = []

    def route(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple]:
        """Directed links from a to b, X first then Y."""
        (x, y), (bx, by) = a, b
        links = []
        while x != bx:
            nx = x + (1 if bx > x else -1)
            links.append(((x, y), (nx, y)))
            x = nx
        while y != by:
            ny = y + (1 if by > y else -1)
            links.append(((x, y), (x, ny)))
            y = ny
        return links

    def send(self, a, b, nbytes: float, t0_ns: float = 0.0) -> int:
        """Queue a burst for this step. Returns its flit count."""
        flits = int(-(-nbytes // self.flit_bytes)) if nbytes > 0 else 0
        if flits:
            self._bursts.append((t0_ns, tuple(a), tuple(b), flits))
            self.total_bytes += nbytes
        return flits

    def deliver(self) -> dict:
        """Run the event queue for the bursts of this step.

        Returns the time the last burst arrives, flit-hops moved and the peak
        load on any one link during the step.
        """
        free_at: dict[tuple, float] = defaultdict(float)
        step_load: dict[tuple, int] = defaultdict(int)
        queue = [(t0, i, 0) for i, (t0, *_rest) in enumerate(self._bursts)]
        heapq.heapify(queue)
        paths = [self.route(a, b) for (_t, a, b, _f) in self._bursts]
        last, flit_hops = 0.0, 0
        while queue:
            t, i, hop = heapq.heappop(queue)
            flits = self._bursts[i][3]
            if hop == len(paths[i]):
                last = max(last, t)
                continue
            link = paths[i][hop]
            start = max(t, free_at[link])            # wait if the link is busy
            done = start + flits * self.t_hop
            free_at[link] = done
            step_load[link] += flits
            self.link_flits[link] += flits
            flit_hops += flits
            heapq.heappush(queue, (done, i, hop + 1))
        self.total_flit_hops += flit_hops
        self._bursts.clear()
        return {"arrival_ns": last, "flit_hops": flit_hops, "peak_link_flits": max(step_load.values(), default=0)}

    def hottest_links(self, k: int = 5) -> list[dict]:
        top = sorted(self.link_flits.items(), key=lambda kv: -kv[1])[:k]
        return [{"from": list(a), "to": list(b), "flits": n} for (a, b), n in top]
