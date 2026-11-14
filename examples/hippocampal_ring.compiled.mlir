// NeuroGraph IR v1
module @hippocampal_continuous_attractor_ring {
  %velocity = ral.evt.encoder {name = "angular velocity cells", population = 64, fan_out = 256, rate = 0.08} : () -> spikes<64, 1> @NeuromorphicCore
  %ring = ral.dyn.attractor {name = "head-direction ring", plasticity = "local_stdp"} (%velocity) : (spikes) -> spikes<256, 8> @NeuromorphicCore
  %heading = ral.evt.decoder {name = "population vector", population = 256} (%ring) : (spikes) -> tensor<2xf32> @TensorAccelerator
  %wrap = ral.geo.project {name = "project to circle", manifold = "torus", dim = 2, composed = 2} (%heading) : (tensor) -> point<torus, 2> @TensorAccelerator
  %path = ral.dyn.koopman {name = "Koopman path predictor", observables = 128} (%wrap) : (tensor) -> tensor<128xi8> @PIMCrossbar
  %place = ral.tensor.readout {name = "place readout", plasticity = "global_sync", in_dim = 128, out_dim = 16} (%path) : (tensor) -> tensor<16xf32> @TensorAccelerator
}
