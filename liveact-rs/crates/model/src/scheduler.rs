//! Neighbor-Forcing flow-matching scheduler (generate.py lines 347-374).
//!
//! SoulX-LiveAct generates autoregressively in 14-latent-frame chunks with a
//! **3-step diffusion schedule** `[1000, 937.5, 833.33, 0]`. The critical
//! property ("Neighbor Forcing"): each diffusion step keeps its OWN KV cache
//! built from step-consistent latents, and the step transition is
//!
//! ```text
//! x0_pred = x_t - t * v_pred              (velocity -> x0)
//! x_{t_next} = (1 - t_next) * x0_pred + eps * t_next   (fresh noise at the
//!                                                        next noise level)
//! ```
//!
//! so every chunk at step i starts from a latent at exactly the noise level
//! the previous chunk's step-i cache saw — step-consistent AR generation
//! with no drift, and constant per-step memory (ConvKV).

use crate::config::LiveActConfig;

pub struct NeighborForcing {
    /// timesteps in [0, 1000]; last entry must be 0 (final x0 readout)
    pub timesteps: Vec<f32>,
    /// diffusion steps that apply audio conditioning (default [1, 2])
    pub audio_steps: Vec<usize>,
}

impl NeighborForcing {
    pub fn new(cfg: &LiveActConfig) -> Self {
        NeighborForcing {
            timesteps: cfg.timesteps.clone(),
            audio_steps: cfg.audio_steps.clone(),
        }
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len() - 1
    }

    #[inline]
    pub fn audio_on(&self, step: usize) -> bool {
        self.audio_steps.contains(&step)
    }

    /// One step transition. Returns the next latent and the x0 prediction
    /// RMS (a useful live metric of denoising progress).
    pub fn step(
        &self,
        latent: &[f32],
        v_pred: &[f32],
        step: usize,
        eps: &[f32],
    ) -> (Vec<f32>, f32) {
        let t = self.timesteps[step] / 1000.0;
        let t_next = self.timesteps[step + 1] / 1000.0;
        debug_assert_eq!(latent.len(), v_pred.len());
        let mut next = vec![0.0f32; latent.len()];
        let mut x0_rms = 0.0f32;
        for i in 0..latent.len() {
            let x0 = latent[i] - v_pred[i] * t;
            x0_rms += x0 * x0;
            next[i] = (1.0 - t_next) * x0 + eps[i] * t_next;
        }
        x0_rms = (x0_rms / latent.len() as f32).sqrt();
        (next, x0_rms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_step_is_x0() {
        let cfg = LiveActConfig::tiny();
        let nf = NeighborForcing::new(&cfg);
        // last transition (t_next = 0): next == x0 exactly
        let latent = vec![1.0, -2.0, 3.0];
        let v = vec![0.5, 0.5, 0.5];
        let eps = vec![9.0, 9.0, 9.0];
        let (next, _) = nf.step(&latent, &v, nf.num_steps() - 1, &eps);
        for (n, l) in next.iter().zip(&latent) {
            let x0 = l - 0.5 * (833.33333333 / 1000.0);
            assert!((n - x0).abs() < 1e-5);
        }
    }

    #[test]
    fn audio_only_on_steps_1_2() {
        let cfg = LiveActConfig::tiny();
        let nf = NeighborForcing::new(&cfg);
        assert!(!nf.audio_on(0));
        assert!(nf.audio_on(1));
        assert!(nf.audio_on(2));
        assert_eq!(nf.num_steps(), 3);
    }

    #[test]
    fn renoise_matches_reference_formula() {
        // x_{t'} = (1-t')*x0 + eps*t'   with  x0 = x_t - t*v
        let t = 0.9375f32;
        let tp = 0.8333333f32;
        let (x, v, e) = (vec![2.0f32], vec![-1.0], vec![0.7]);
        let x0 = x[0] - t * v[0];
        let expect = (1.0 - tp) * x0 + e[0] * tp;
        let cfg = LiveActConfig::tiny();
        let nf = NeighborForcing::new(&cfg);
        // emulate step 1
        let (got, _) = nf.step(&x, &v, 1, &e);
        assert!((got[0] - expect).abs() < 1e-4);
    }
}
