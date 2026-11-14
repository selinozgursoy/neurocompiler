// NeuroGraph IR v1
module @closed_loop_spiking_bci_manifold_decoder {
  %electrodes = ral.evt.encoder {name = "recorded units", population = 256, fan_out = 128, rate = 0.06} : () -> spikes<256, 1>
  %population = ral.evt.lif {name = "adaptive population", plasticity = "local_stdp", population = 512, fan_out = 128, rate = 0.04} (%electrodes, %stimulation feedback) : (spikes, spikes) -> spikes<512, 1>
  %latent = ral.evt.decoder {name = "latent decoder", population = 512, decode_dim = 16} (%population) : (spikes) -> tensor<16xf32>
  %manifold = ral.geo.project {name = "project to sphere", manifold = "sphere", dim = 16} (%latent) : (tensor) -> point<sphere, 16>
  %dynamics = ral.dyn.koopman {name = "Koopman latent dynamics", observables = 96} (%manifold) : (tensor) -> tensor<96xi8>
  %controller = ral.tensor.mlp {name = "controller", plasticity = "global_sync", in_dim = 96, out_dim = 32, layers = 2} (%dynamics) : (tensor) -> tensor<32xf32>
  %stimulation = ral.evt.stim {name = "stimulation encoder", fan_out = 64, rate = 0.03} (%controller) : (tensor) -> spikes<128, 8>
}
