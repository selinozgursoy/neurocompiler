"""Functional kernels: what each IR node computes during simulation.

A kernel returns its output, the operation counts of the step (which the cost
model turns into energy and latency for the node's target) and metrics.
Stimuli and weights are drawn from a generator seeded by (seed, node id), so
the heterogeneous run and the GPU baseline see identical inputs.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from ..compiler.cost import Ops
from ..compiler.hardware import HardwareConfig
from ..geometry import (equivariance_error, intertwiner, parse_irreps, random_rotation, rep_matrix,
                        sph_harm_features, wigner)
from ..ir import (ContinuousAttractor, DenseTensorOp, KoopmanOperator, ManifoldGroupAction, Node,
                  Plasticity, SpikeEventNode, Target)


def node_rng(seed: int, nid: str, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(nid.encode()), salt])


class Kernel:
    def __init__(self, node: Node, hw: HardwareConfig, seed: int):
        self.node, self.hw, self.seed = node, hw, seed
        self.rng = node_rng(seed, node.id)          # weights and stimuli: identical across modes
        self.noise = node_rng(seed, node.id, 1)     # device noise: only used on analog targets
        self.f = node.features()
        self._fits: dict[tuple[int, int, int], np.ndarray] = {}
        self.info: dict = {}

    @property
    def target(self) -> Target:
        return self.node.target

    def fit(self, x: np.ndarray, dim: int, slot: int = 0) -> np.ndarray:
        """Adapt an input of any length to `dim` with a fixed random projection."""
        x = np.asarray(x, float).ravel()
        if x.size == dim:
            return x
        key = (slot, x.size, dim)
        if key not in self._fits:
            self._fits[key] = node_rng(self.seed, self.node.id, 100 + slot).normal(size=(dim, x.size)) / math.sqrt(x.size)
        return self._fits[key] @ x

    def merged(self, inputs: list[np.ndarray], dim: int) -> np.ndarray:
        if not inputs:
            return np.zeros(dim)
        return sum(self.fit(x, dim, i) for i, x in enumerate(inputs))

    def step(self, t: int, inputs: list[np.ndarray]):
        raise NotImplementedError

    def consolidate(self) -> int:
        """Merge the fast delta into the slow weights. Returns synapses shipped."""
        return 0


# --------------------------------------------------------------------------- events

class EncoderKernel(Kernel):
    """Tensor -> spikes. As a source it encodes a slow latent velocity in two
    opponent sub-populations, so downstream attractors can integrate it."""
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        n = node.population
        self.sign = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        self.latent = 0.0
        self.phase = self.rng.uniform(0, 2 * math.pi)

    def step(self, t, inputs):
        n = self.node
        if inputs:
            drive = np.tanh(self.merged(inputs, n.population))
        else:
            self.latent = 0.7 * math.sin(2 * math.pi * t / 240.0 + self.phase) + 0.3 * math.sin(2 * math.pi * t / 61.0)
            drive = self.sign * self.latent
        p = np.clip(n.rate * (1.0 + drive), 0.0, 1.0)
        spikes = (self.rng.random(n.population) < p).astype(float)
        ev = float(spikes.sum())
        return spikes, Ops(macs=self.f.macs, synops=ev * max(1, n.fan_out), neuron_updates=n.population, events=ev), {}


class LIFKernel(Kernel):
    """Leaky integrate-and-fire population with optional pair-based STDP."""
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        self.n = node.population
        self.v = np.zeros(self.n)
        self.th = np.ones(self.n)
        self.W: list[np.ndarray] = []
        self.mask: list[np.ndarray] = []
        self.W_slow: list[np.ndarray] = []
        self.pre: list[np.ndarray] = []
        self.post = np.zeros(self.n)
        self.plastic = node.plasticity == Plasticity.LOCAL_STDP

    def _weights(self, i: int, m: int):
        while len(self.W) <= i:
            fan_in = min(m, max(1, self.node.fan_out))
            mask = np.zeros((self.n, m), bool)
            for r in range(self.n):
                mask[r, self.rng.choice(m, size=fan_in, replace=False)] = True
            W = np.where(mask, self.rng.uniform(0.2, 0.6, size=(self.n, m)), 0.0) * (4.0 / fan_in)
            self.W.append(W); self.mask.append(mask); self.W_slow.append(W.copy()); self.pre.append(np.zeros(m))
        return self.W[i]

    def step(self, t, inputs):
        n = self.node
        cur = np.zeros(self.n)
        synops = stdp = 0.0
        for i, s in enumerate(inputs):
            s = np.asarray(s, float).ravel()
            W = self._weights(i, s.size)
            cur += W @ s
            nz = s > 0
            synops += float(self.mask[i][:, nz].sum())        # one op per synapse an input spike reaches
            self.pre[i] = 0.9 * self.pre[i] + s
        self.v = 0.9 * self.v + cur + 0.02
        spikes = (self.v > self.th).astype(float)
        self.v[spikes > 0] = 0.0
        self.th += 0.02 * (spikes - n.rate)                   # homeostasis keeps the declared rate
        self.th = np.clip(self.th, 0.2, 5.0)
        if self.plastic:
            for i, s in enumerate(inputs):
                s = np.asarray(s, float).ravel()
                post_rows = spikes > 0
                pre_cols = s > 0
                W, mask = self.W[i], self.mask[i]
                W[post_rows] += 2e-3 * self.pre[i][None, :] * mask[post_rows]        # pre before post: potentiate
                W[:, pre_cols] -= 2e-3 * self.post[:, None] * mask[:, pre_cols]      # post before pre: depress
                np.clip(W, 0.0, 1.0, out=W)
                stdp += float(mask[post_rows].sum() + mask[:, pre_cols].sum())
        self.post = 0.9 * self.post + spikes
        ev = float(spikes.sum())
        drift = 0.0
        if self.plastic and self.W:
            num = math.sqrt(sum(float(((w - s) ** 2).sum()) for w, s in zip(self.W, self.W_slow)))
            den = math.sqrt(sum(float((s ** 2).sum()) for s in self.W_slow)) or 1.0
            drift = num / den
        dense = float(sum(m.sum() for m in self.mask)) if self.mask else self.f.macs
        return spikes, Ops(macs=dense, synops=synops, neuron_updates=self.n, stdp_updates=stdp, events=ev), {"sync_drift": drift}

    def consolidate(self):
        shipped = 0
        for i, w in enumerate(self.W):
            shipped += int((np.abs(w - self.W_slow[i]) > 1e-9).sum())
            self.W_slow[i] = w.copy()
        return shipped


class DecoderKernel(Kernel):
    """Spikes -> tensor. With two outputs it is a population-vector readout."""
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        self.filt = None
        self.D = None

    def step(self, t, inputs):
        n = self.node
        s = np.asarray(inputs[0], float).ravel() if inputs else np.zeros(n.population)
        if self.D is None:
            if n.decode_dim == 2:
                th = np.linspace(0, 2 * math.pi, s.size, endpoint=False)
                self.D = np.vstack([np.cos(th), np.sin(th)])
            else:
                self.D = self.rng.normal(size=(n.decode_dim, s.size)) / math.sqrt(s.size)
            self.filt = np.zeros(s.size)
        graded = bool(inputs) and (s.max(initial=0.0) > 1.0 or (s % 1).any())
        self.filt = s if graded else 0.8 * self.filt + 0.2 * s      # graded events already carry rates
        out = self.D @ self.filt
        ev_in = float((s != 0).sum())
        return out, Ops(macs=float(self.D.size), synops=ev_in * n.decode_dim, neuron_updates=n.decode_dim, events=0.0), {}


class StimKernel(EncoderKernel):
    """Tensor -> spikes for stimulation: a rate code of the control signal."""


# ----------------------------------------------------------------------- attractor

class AttractorKernel(Kernel):
    """Threshold-linear ring attractor. On a neuromorphic core it uses
    sigma-delta coding: a neuron transmits only when its rate has moved by a
    threshold, and the recurrence sees the transmitted values."""
    J0, J1, I0, TAU, DT = -2.0, 6.0, 1.0, 0.010, 0.001
    SPEED_PER_GAMMA = 16.667     # rad/s of bump motion per unit of asymmetric coupling

    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        N = self.N = node.neurons
        self.th = np.linspace(0, 2 * math.pi, N, endpoint=False)
        d = self.th[:, None] - self.th[None, :]
        self.Wsym = (self.J0 + self.J1 * np.cos(d)) / N
        self.Wasym = np.sin(d) / N
        self.dW = np.zeros((N, N))               # fast, local, plastic delta
        self.plastic = node.plasticity == Plasticity.LOCAL_STDP
        self.event_driven = self.target == Target.NEUROMORPHIC
        r = np.maximum(0, np.cos(self.th - math.pi)) + 0.01
        for _ in range(1500):                    # settle onto the attractor
            r += self.DT / self.TAU * (-r + np.maximum(0, self.Wsym @ r + self.I0))
        self.r, self.sent = r, r.copy()
        self.delta = 0.05 * float(r.max())
        self.peak0 = float(r.max())
        self.phi_ideal = self._phase(r)
        self.v_est = 0.0
        self.tot = 0.0
        self.gain = 0.4

    @staticmethod
    def _phase_of(r, th):
        return float(np.angle(np.sum(r * np.exp(1j * th))))

    def _phase(self, r):
        return self._phase_of(r, self.th)

    def step(self, t, inputs):
        # Velocity estimate from opponent spike counts (or a tensor drive).
        if inputs:
            s = np.asarray(inputs[0], float).ravel()
            half = s.size // 2
            self.tot = 0.98 * self.tot + 0.02 * float(s.sum())      # running event count
            raw = (s[half:].sum() - s[:half].sum()) / max(1.0, self.tot)
            self.v_est = 0.9 * self.v_est + 0.1 * float(np.clip(raw, -1.5, 1.5))
            fb = sum(float(np.mean(x)) for x in inputs[1:]) if len(inputs) > 1 else 0.0
            self.v_est += 0.02 * math.tanh(fb)
        gamma = self.gain * self.v_est
        W = self.Wsym + gamma * self.Wasym + self.dW
        drive = self.sent if self.event_driven else self.r
        self.r = self.r + self.DT / self.TAU * (-self.r + np.maximum(0, W @ drive + self.I0))
        fire = np.abs(self.r - self.sent) >= self.delta
        ev = float(fire.sum())
        if self.event_driven:
            self.sent = np.where(fire, self.r, self.sent)
        else:
            self.sent = self.r.copy()
        stdp = 0.0
        if self.plastic:
            # Local Hebbian drift: every synapse sees its own pre/post activity plus noise.
            rn = self.r / (self.peak0 or 1.0)
            self.dW += 2e-6 * (np.outer(rn, rn) - rn.mean() ** 2) + 2e-5 * self.noise.normal(size=self.dW.shape) / self.N
            stdp = ev * self.N if self.event_driven else float(self.N * self.N)
        self.phi_ideal += self.SPEED_PER_GAMMA * gamma * self.DT
        err = math.remainder(self._phase(self.r) - self.phi_ideal, 2 * math.pi)
        drift = float(np.linalg.norm(self.dW) / np.linalg.norm(self.Wsym))
        ops = Ops(macs=float(self.N * self.N), synops=ev * (self.node.fan_out or self.N), neuron_updates=self.N,
                  stdp_updates=stdp, events=ev)
        return self.sent.copy(), ops, {"phase_error": abs(err), "bump_amplitude": float(self.r.max() / self.peak0),
                                       "sync_drift": drift}

    def consolidate(self):
        """Project the learned delta back onto the ring's symmetry (circulant
        matrices), fold it into the slow weights and clear the fast delta."""
        if not self.plastic:
            return 0
        N = self.N
        shipped = int((np.abs(self.dW) > 0).sum())
        idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
        diag_mean = np.bincount(idx.ravel(), weights=self.dW.ravel(), minlength=N) / N
        self.Wsym = self.Wsym + 0.1 * diag_mean[idx]         # alpha = 0.1, circulant part only
        self.dW[:] = 0.0
        return shipped


# ------------------------------------------------------------------------ geometry

class GeoKernel(Kernel):
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        n = node
        self.kind = n.kind
        self.on_pim = self.target == Target.PIM
        self.R = np.eye(3)
        if self.kind in ("project", "exp_map", "log_map"):
            return
        self.irr_out = parse_irreps(n.irreps_out or n.irreps_in)
        self.irr_in = parse_irreps(n.irreps_in or n.irreps_out)
        if max(i.l for i in self.irr_in + self.irr_out) > 2:
            raise ValueError(f"%{n.id}: numeric simulation supports irreps with l <= 2")
        if self.kind == "sph_harm":
            self.coef = [self.rng.normal(size=i.mul) for i in self.irr_out]
            self.axis = random_rotation(self.rng)
        if self.kind in ("tensor_product", "fused_conv"):
            self.irr_sh = parse_irreps(n.irreps_sh or "1x0e+1x1o+1x2e")
            self.paths = []
            for a, ia in enumerate(self.irr_in):
                for ib in self.irr_sh:
                    for c, ic in enumerate(self.irr_out):
                        C = intertwiner(ia.l, ib.l, ic.l)
                        if C is not None:
                            Wp = self.rng.normal(size=(ic.mul, ia.mul)) / math.sqrt(ia.mul * 3)
                            self.paths.append((a, ib.l, c, C, Wp))
        if self.kind in ("equiv_linear", "fused_conv"):
            src = self.irr_out if self.kind == "fused_conv" else self.irr_in
            self.blocks = []
            for o in self.irr_out:
                for i_idx, i in enumerate(src):
                    if i.l == o.l:
                        W = self.rng.normal(size=(o.mul, i.mul)) / math.sqrt(i.mul)
                        if self.on_pim:      # programmed once: one error per physical cell
                            W = W + hw.pim.prog_noise * np.abs(W).max() * self.noise.normal(size=W.shape)
                        self.blocks.append((self.irr_out.index(o), i_idx, W))

    @staticmethod
    def _split(x, irr):
        parts, o = [], 0
        for i in irr:
            d = 2 * i.l + 1
            parts.append(x[o:o + i.mul * d].reshape(i.mul, d))
            o += i.mul * d
        return parts

    def _linear(self, parts_in, src_irr):
        out = [np.zeros((o.mul, 2 * o.l + 1)) for o in self.irr_out]
        pim = self.hw.pim
        for oi, ii, W in self.blocks:
            X = parts_in[ii]
            if self.on_pim:          # one read per component, each with fresh read noise
                scale = pim.read_noise * np.abs(W).max()
                Y = np.stack([(W + scale * self.noise.normal(size=W.shape)) @ X[:, c] for c in range(X.shape[1])], axis=1)
            else:
                Y = W @ X
            out[oi] += Y
        return out

    def step(self, t, inputs):
        n, k = self.node, self.kind
        m = {}
        if k in ("project", "exp_map", "log_map"):
            x = self.merged(inputs, n.dim) if inputs else self.rng.normal(size=n.dim)
            if k == "project":
                if n.manifold == "so3" and n.dim == 9:
                    U, _, Vt = np.linalg.svd(x.reshape(3, 3))
                    Rm = U @ Vt
                    if np.linalg.det(Rm) < 0:
                        U[:, -1] *= -1
                        Rm = U @ Vt
                    y = Rm.ravel()
                    m["manifold_residual"] = float(np.linalg.norm(Rm @ Rm.T - np.eye(3)))
                elif n.manifold == "torus":
                    pairs = x[: (x.size // 2) * 2].reshape(-1, 2)
                    norm = np.linalg.norm(pairs, axis=1, keepdims=True)
                    y = np.concatenate([(pairs / np.where(norm > 1e-9, norm, 1.0)).ravel(), x[(x.size // 2) * 2:]])
                else:
                    y = x / (np.linalg.norm(x) or 1.0)
                    m["manifold_residual"] = abs(float(np.linalg.norm(y)) - 1.0)
            elif k == "exp_map":      # sphere, base point e_0, tangent = x with x[0] dropped
                v = x.copy(); v[0] = 0.0
                a = np.linalg.norm(v)
                y = np.zeros_like(x); y[0] = math.cos(a)
                if a > 1e-12:
                    y += math.sin(a) * v / a
            else:                     # log_map, inverse of the above
                p = x / (np.linalg.norm(x) or 1.0)
                a = math.acos(float(np.clip(p[0], -1, 1)))
                v = p.copy(); v[0] = 0.0
                nv = np.linalg.norm(v)
                y = a * v / nv if nv > 1e-12 else np.zeros_like(x)
            return y, Ops(macs=self.f.macs), m

        dim_in = sum(i.dim for i in self.irr_in)
        if k == "sph_harm":
            ang = 2 * math.pi * t / 300.0
            p = self.axis @ np.array([math.cos(ang), math.sin(ang), 0.3 * math.sin(2 * ang)])
            if inputs:
                p = p + 0.1 * self.fit(inputs[0], 3)
            Y = sph_harm_features(p)
            y = np.concatenate([np.outer(c, Y[i.l]).ravel() for c, i in zip(self.coef, self.irr_out)])
            return y, Ops(macs=self.f.macs), m

        x = self.merged(inputs, dim_in) if inputs else self.rng.normal(size=dim_in)
        parts = self._split(x, self.irr_in)
        if k == "group_act":
            self.R = self.R @ _small_rotation(0.01 * n.composed)
            y = np.concatenate([(p @ wigner(i.l, self.R).T).ravel() for p, i in zip(parts, self.irr_in)])
            return y, Ops(macs=self.f.macs), m
        if k in ("tensor_product", "fused_conv"):
            ang = 2 * math.pi * t / 300.0
            Y = sph_harm_features(np.array([math.cos(ang), math.sin(ang), 0.2]))
            out = [np.zeros((o.mul, 2 * o.l + 1)) for o in self.irr_out]
            for a, lb, c, C, Wp in self.paths:
                out[c] += Wp @ np.einsum("mi,j,ijk->mk", parts[a], Y[lb], C)
            if k == "fused_conv":
                out = self._linear(out, self.irr_out)
            y = np.concatenate([o.ravel() for o in out])
            return y / (1.0 + np.linalg.norm(y) / math.sqrt(y.size)), Ops(macs=self.f.macs), m
        # equiv_linear
        out = self._linear(parts, self.irr_in)
        y = np.concatenate([o.ravel() for o in out])
        if t % 16 == 0 and n.irreps_in == (n.irreps_out or n.irreps_in):
            m["symmetry_error"] = self._symmetry_error(x, y, t)
        return y, Ops(macs=self.f.macs), m

    def _symmetry_error(self, x, y, t):
        """|| f(D x) - D f(x) || / || f(x) || for one random rotation."""
        R = random_rotation(node_rng(self.seed, self.node.id, 7000 + t))
        D = rep_matrix(self.node.irreps_in, R)
        fy = np.concatenate([o.ravel() for o in self._linear(self._split(D @ x, self.irr_in), self.irr_in)])
        return float(np.linalg.norm(fy - D @ y) / (np.linalg.norm(y) or 1.0))


def _small_rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------------- koopman

class KoopmanKernel(Kernel):
    """z' = K z + B u. On a crossbar K is quantized, programmed with error and
    read with noise; a float shadow trajectory measures the deviation."""
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        n = self.n = node.observables
        K = np.zeros((n, n))
        for b in range(n // 2):                      # stable rotation blocks
            rad = node.rho * self.rng.uniform(0.85, 1.0)
            ang = self.rng.uniform(0.05, 0.6)
            K[2 * b:2 * b + 2, 2 * b:2 * b + 2] = rad * np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        if n % 2:
            K[-1, -1] = node.rho * 0.9
        Q, _ = np.linalg.qr(self.rng.normal(size=(n, n)))
        self.K = Q @ K @ Q.T
        self.on_pim = self.target == Target.PIM
        self.Kdev = self.K
        if self.on_pim:
            pim = hw.pim
            lim = np.abs(self.K).max()
            step = 2 * lim / (2 ** pim.bits - 1)
            self.Kdev = np.round(self.K / step) * step + pim.prog_noise * lim * self.noise.normal(size=(n, n))
            self.read_scale = pim.read_noise * lim
        self.info["rho_float"] = float(np.abs(np.linalg.eigvals(self.K)).max())
        self.info["rho_device"] = float(np.abs(np.linalg.eigvals(self.Kdev)).max())
        self.z = np.zeros(n)
        self.z_ref = np.zeros(n)

    def step(self, t, inputs):
        n = self.node
        u = 0.2 * np.tanh(self.merged(inputs, self.n)) if inputs else 0.2 * np.sin(0.05 * t + np.arange(self.n))
        for _ in range(n.steps):
            self.z_ref = self.K @ self.z_ref + u
            if self.on_pim:
                self.z = (self.Kdev + self.read_scale * self.noise.normal(size=self.K.shape)) @ self.z + u
            else:
                self.z = self.K @ self.z + u
        if self.on_pim and n.reencode_every and (t + 1) % n.reencode_every == 0:
            self.z = self.z_ref.copy()                  # re-lift from the true state
        dev = float(np.linalg.norm(self.z - self.z_ref) / (np.linalg.norm(self.z_ref) or 1.0))
        return self.z.copy(), Ops(macs=self.f.macs), {"koopman_deviation": dev}


# --------------------------------------------------------------------------- dense

class DenseKernel(Kernel):
    def __init__(self, node, hw, seed):
        super().__init__(node, hw, seed)
        n = node
        self.Ws = []
        if n.kind in ("matmul", "readout", "reduce"):
            self.Ws = [self.rng.normal(size=(n.out_dim, n.in_dim)) / math.sqrt(n.in_dim)]
        elif n.kind == "mlp":
            dims = [n.in_dim] + [max(n.in_dim, n.out_dim)] * (max(1, n.layers) - 1) + [n.out_dim]
            self.Ws = [self.rng.normal(size=(b, a)) / math.sqrt(a) for a, b in zip(dims[:-1], dims[1:])]
        self.learns = n.plasticity == Plasticity.GLOBAL_SYNC

    def step(self, t, inputs):
        n = self.node
        x = self.merged(inputs, n.in_dim) if inputs else self.rng.normal(size=n.in_dim)
        macs = self.f.macs
        if n.kind == "nonlinearity":
            y = np.tanh(x)
        else:
            y = x
            for i, W in enumerate(self.Ws):
                y = W @ y
                if n.kind == "mlp" and i < len(self.Ws) - 1:
                    y = np.tanh(y)
            if self.learns:        # a global update of the last layer on the accelerator
                self.Ws[-1] *= (1.0 - 1e-4)
                macs += float(self.Ws[-1].size)
        return y, Ops(macs=macs), {}


def make_kernel(node: Node, hw: HardwareConfig, seed: int) -> Kernel:
    if isinstance(node, SpikeEventNode):
        return {"encoder": EncoderKernel, "lif": LIFKernel, "decoder": DecoderKernel, "stim": StimKernel}[node.kind](node, hw, seed)
    if isinstance(node, ContinuousAttractor):
        return AttractorKernel(node, hw, seed)
    if isinstance(node, ManifoldGroupAction):
        return GeoKernel(node, hw, seed)
    if isinstance(node, KoopmanOperator):
        return KoopmanKernel(node, hw, seed)
    if isinstance(node, DenseTensorOp):
        return DenseKernel(node, hw, seed)
    raise TypeError(f"no kernel for {type(node).__name__}")
