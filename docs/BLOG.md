# Compiling neural circuit geometry onto heterogeneous silicon

*A walk through NeuroGraph-HW: what it is, how it works, and what its simulator says.*

A grid-cell module keeps its population activity on a torus. A molecular-dynamics network keeps its features in irreducible representations of O(3). A Koopman autoencoder turns a nonlinear system into one matrix multiply. In each of these models the structure is the point, and a conventional compiler throws it away before the first instruction reaches hardware.

NeuroGraph-HW keeps it. It is a typed graph IR, a compiler that partitions a circuit across three kinds of hardware, and a runtime that simulates the result. This post explains the design and then reports what the simulator found, including the cases where heterogeneity loses.

## Why one kind of hardware fits badly

Three workloads, three mismatches with a dense accelerator.

**Block-sparse kernels.** The Clebsch-Gordan tensor product at the heart of rotation-equivariant networks is 75% to 90% zeros. Dense libraries pad the zeros back in.

**Idle work.** A ring attractor holds a bump of activity. When the bump sits still nothing changes, yet a dense accelerator multiplies the full recurrent matrix every step.

**Learning that cannot stay local.** Synaptic plasticity updates a weight from the activity on either side of it. On a von Neumann machine that is a round trip to memory for every synapse.

Hardware that fits each case exists: asynchronous neuromorphic cores that pay per event, analog crossbars that multiply inside the memory that stores the matrix, and dense accelerators for everything nonlinear. What is missing is a compiler that can tell the cases apart.

## The IR: geometry is a type

The IR has five node classes. `ManifoldGroupAction` covers equivariant operators and manifold maps. `KoopmanOperator` is a frozen linear step. `ContinuousAttractor` is a ring or torus network. `SpikeEventNode` covers populations and codecs. `DenseTensorOp` is everything else.

Edges come in two kinds: `tensor` for continuous data and `spikes` for sparse address-event packets. A value never changes kind silently. If spikes need to reach a tensor op, the graph must contain an explicit decoder, so the cost of that crossing is visible to the partitioner.

Types carry what passes need. A feature field is `rep<SO3, "32x0e+16x1o+8x2e">`. The verifier checks tensor products against the selection rules, so a rewrite that breaks equivariance produces invalid IR.

## Fusion: what the geometry lets you delete

Because the compiler knows the algebra, it can remove work exactly:

- Projecting twice onto the same manifold is projecting once.
- An exponential map followed by its logarithm is nothing.
- Two group actions compose into one.
- Repeated Koopman steps fold into one operator, K^k.
- A tensor product followed by an equivariant linear map becomes one fused op.

The last rewrite is Schur's lemma. An equivariant linear map is block diagonal by irrep and repeats one small matrix across the 2l+1 components of each block. Store that matrix once and read it 2l+1 times. On the SE(3) preset, fusion takes 12 nodes to 8 and halves the edges that cross between targets.

Schur sharing has a second effect that we did not expect. If the same physical cells serve every component, a programming error E gives (W + E) ⊗ I, which still commutes with the group action. Static device variability cannot break the symmetry. The test suite checks this end to end: with 5% programming error and no read noise, the measured equivariance error is below 10⁻⁹.

## Partitioning: a cost model, not a rule table

Each node gets an energy estimate on each target and a feasibility verdict.

- A neuromorphic core pays per synaptic operation, per neuron update and per STDP update. Cost follows activity.
- A crossbar pays mostly for analog-to-digital conversions. It accepts only linear, 8-bit, non-plastic maps.
- The accelerator pays per multiply-accumulate, and pays again for DRAM traffic when a weight matrix does not fit in cache.
- An edge that crosses targets pays for mesh hops, for the bridge, and for transcoding between events and tensors.

The partitioner seeds each node on its cheapest feasible target, refines one node at a time while the objective falls, and evicts the least-regretted nodes if cores or tiles run out.

One thing is a contract and not a cost. A node marked `local_stdp` has its weight written by an edge core's learning engine, so it lives on a core. That decision has consequences, as the results show.

## Runtime: local learning, global reconciliation

Two parties want to change a plastic weight: the core, on every spike, and the optimizer, from a global loss. The runtime never lets both write.

The core owns a fast delta. The accelerator owns the slow weights. Global learning reaches the core as a modulator, one value per neuron, broadcast over the mesh. That is the third factor of a three-factor rule, and it means downstream traffic grows with neurons and not with synapses.

At each epoch barrier the delta is shipped, projected back onto the circuit's symmetry, and committed. For the ring attractor that projection is onto circulant matrices: local learning knows nothing about translation invariance, so consolidation is where it gets restored.

Feedback edges are epoch-registered: a reader sees only what was written in the previous epoch. That one rule breaks every wait cycle between the mesh and the accelerator.

## What the simulator found

Four hundred steps, default constants, identical stimuli for both mappings.

| Preset | Energy | Latency | Weight bytes moved |
| --- | --- | --- | --- |
| SE(3) equivariant spatial circuit | 1.24× lower | 72 µs vs 80 µs | 51% fewer |
| Hippocampal attractor ring | 3.22× lower | 35 µs vs 60 µs | 95% fewer |
| Closed-loop BCI decoder | 2.40× higher | 108 µs vs 70 µs | 90% fewer |

**Sparse dynamics are the win.** The ring is event-driven and mostly quiet. The break-even sits near 10% activity for a 256-neuron ring; above it the accelerator is cheaper.

**On-core plasticity is the loss.** Published figures put a synaptic update at 120 pJ on Loihi. The BCI preset spends 82% of its energy there and ends up 2.4× worse than an accelerator at peak efficiency. Keeping learning local removes the memory round trip, but it does not come free.

**Accuracy moves too.** Event coding costs the ring 0.10 rad of RMS phase error against 0.002 for the dense baseline. A Koopman operator on a crossbar with 1% device noise drifts 25% to 41% from its float trajectory even when re-encoded every eight steps. A real deployment would need quieter devices or a shorter re-encode period.

**Scale changes the picture.** While weights fit in the accelerator's cache, the ring's advantage is 1.7× to 3.5×. Once they do not, the accelerator streams them from DRAM every step and the model's ratio jumps by two orders of magnitude. That is the memory wall, and it is where in-memory and event-driven hardware earn their keep.

## How much to trust this

Not as a prediction. The simulator computes behaviour and estimates energy from constants that come from different chips and process nodes: a 2018 neuromorphic chip, a 2023 crossbar chip, an accelerator datasheet. It models dynamic energy only, and it charges the accelerator its best-case efficiency. Static power, which dominates small real-time systems, is absent on every target.

What the simulator is good for is the shape of the trade: which term dominates, where the break-even sits, what a design choice costs in accuracy. Every constant is editable, in the studio or from the command line, so you can ask the same questions with your own numbers.

## Try it

```bash
docker compose up --build      # http://localhost:7860
neurograph benchmark hippocampal_ring
```

The geometry in these models is information the hardware can use. A compiler should keep it long enough to matter, and should say plainly when using it does not pay.
