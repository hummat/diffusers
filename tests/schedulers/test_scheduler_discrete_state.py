import torch

from diffusers import DiscreteStateScheduler, DDPMScheduler

from .test_schedulers import SchedulerCommonTest


class DiscreteStateSchedulerTest(SchedulerCommonTest):

    def test_add_noise(self):
        scheduler = DiscreteStateScheduler(beta_schedule="linear",
                                           rescale_betas_zero_snr=True)

        original_samples = torch.randint(0, scheduler.config.num_classes, (10, 1, 32 ** 3))
        noise = torch.rand(*original_samples.shape, scheduler.config.num_classes)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (original_samples.size(0),))

        simple_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="simple")
        assert original_samples.size() == simple_noisy.size()
        assert torch.any(simple_noisy != 0)
        assert not torch.allclose(simple_noisy, original_samples)

        log_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="log")
        assert original_samples.size() == log_noisy.size()
        assert torch.equal(simple_noisy, log_noisy), f"{simple_noisy.sum()}, {log_noisy.sum()}"

        matrix_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="matrix")
        assert original_samples.size() == matrix_noisy.size()
        assert torch.equal(simple_noisy, matrix_noisy)

    def test_step(self):
        scheduler = DiscreteStateScheduler(beta_schedule="linear",
                                           rescale_betas_zero_snr=True)

        model_output = torch.rand(1, 1, 32 ** 3)
        model_output[model_output < 0.5] *= -1
        sample = torch.randint_like(model_output, 0, scheduler.config.num_classes)

        for t in reversed([0, 1, 2, 10, 100, 500, 999]):
            model_output = torch.rand(1, 1, 32 ** 3)
            model_output[model_output < 0.5] *= -1
            noise = torch.rand(*sample.shape, scheduler.config.num_classes)

            out_simple = scheduler.step(model_output, t, sample, noise, implementation="simple")
            assert out_simple.prev_sample.size() == sample.size()
            assert torch.any(out_simple.prev_sample != 0)
            if t == 0:
                assert torch.equal(out_simple.prev_sample, out_simple.pred_original_sample)
            elif t > 10:
                assert not torch.equal(out_simple.prev_sample, sample)
            assert out_simple.pred_original_sample.size() == model_output.size()
            assert torch.any(out_simple.pred_original_sample != 0)

            out_log = scheduler.step(model_output, t, sample, noise, implementation="log")
            assert torch.equal(out_simple.prev_sample, out_log.prev_sample)
            assert torch.equal(out_simple.pred_original_sample, out_log.pred_original_sample)

            out_matrix = scheduler.step(model_output, t, sample, noise, implementation="matrix")
            assert torch.equal(out_simple.prev_sample, out_matrix.prev_sample)
            assert torch.equal(out_simple.pred_original_sample, out_matrix.pred_original_sample)

            sample = out_simple.prev_sample
