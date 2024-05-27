import torch

from diffusers import DiscreteStateScheduler

from .test_schedulers import SchedulerCommonTest


class DiscreteStateSchedulerTest(SchedulerCommonTest):

    def test_add_noise(self):
        scheduler = DiscreteStateScheduler(num_classes=2)

        original_samples = torch.randint(0, scheduler.config.num_classes, (10, 1, 16 ** 3))
        noise = torch.rand(*original_samples.shape, scheduler.config.num_classes)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (original_samples.size(0),))

        simple_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="simple")
        assert original_samples.size() == simple_noisy.size()

        log_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="log")
        assert original_samples.size() == log_noisy.size()
        assert torch.equal(simple_noisy, log_noisy), f"{simple_noisy.sum()}, {log_noisy.sum()}"

        matrix_noisy = scheduler.add_noise(original_samples, noise, timesteps, implementation="matrix")
        assert original_samples.size() == matrix_noisy.size()
        assert torch.equal(simple_noisy, matrix_noisy)

    def test_step(self):
        scheduler = DiscreteStateScheduler(num_classes=2)

        model_output = torch.rand(1, 1, 16 ** 3)
        model_output[model_output < 0.5] *= -1
        sample = torch.randint_like(model_output, 0, scheduler.config.num_classes)

        for t in [0, 1, 2, 10, 100, 500, 999]:
            out_simple = scheduler.step(model_output, t, sample, implementation="simple")
            out_log = scheduler.step(model_output, t, sample, implementation="log")
            # assert torch.equal(out_simple.prev_sample, out_log.prev_sample)
            # assert torch.equal(out_simple.pred_original_sample, out_log.pred_original_sample)

