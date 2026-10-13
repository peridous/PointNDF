import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest
import numpy as np
import torch
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('trainer', CODE / 'train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

class TrainerTests(unittest.TestCase):

    def test_rng_roundtrip(self):
        generator = torch.Generator().manual_seed(51)
        state = trainer.capture_rng(generator)

        def draw():
            return (random.random(), np.random.random(), torch.rand(1).item(), torch.rand(1, generator=generator).item())
        expected = draw()
        trainer.restore_rng(state, generator)
        self.assertEqual(expected, draw())

    def test_epoch_resume_matches_uninterrupted_updates(self):

        def create():
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.8, 0.9))
            scheduler = trainer.make_scheduler(optimizer)
            return (model, optimizer, scheduler)

        def epoch(parts, steps):
            model, optimizer, scheduler = parts
            for _ in range(25):
                optimizer.zero_grad()
                model(torch.ones(3, 2)).square().mean().backward()
                optimizer.step()
                if steps <= 20:
                    scheduler.step()
                steps += 1
            scheduler.step()
            return steps
        parts = create()
        steps = epoch(parts, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pth'
            trainer.atomic_save(dict(model=parts[0].state_dict(), optimizer=parts[1].state_dict(), scheduler=trainer.scheduler_state(parts[2])), path)
            self.assertFalse(path.with_suffix('.pth.tmp').exists())
            saved = torch.load(path, weights_only=False)
            resumed = create()
            resumed[0].load_state_dict(saved['model'])
            resumed[1].load_state_dict(saved['optimizer'])
            trainer.load_scheduler_state(resumed[2], saved['scheduler'])
            self.assertIs(resumed[2].after_scheduler.optimizer, resumed[1])
            for _ in range(22):
                next_steps = epoch(parts, steps)
                self.assertEqual(next_steps, epoch(resumed, steps))
                steps = next_steps
                self.assertEqual(parts[1].param_groups[0]['lr'], resumed[1].param_groups[0]['lr'])
            for first, second in zip(parts[0].parameters(), resumed[0].parameters()):
                self.assertTrue(torch.equal(first, second))

    def test_data_layout_rejects_missing_noisy_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for resolution in trainer.RESOLUTIONS:
                p = root / 'pointclouds/train' / resolution
                p.mkdir(parents=True)
                (p / 'a.xyz').write_text('0 0 0\n')
            clean = root / 'pointclouds/test/clean/10000_poisson'
            clean.mkdir(parents=True)
            (clean / 'a.xyz').write_text('0 0 0\n')
            args = trainer.parse_args(['--data-root', folder, '--output', str(root / 'run')])
            with self.assertRaisesRegex(ValueError, 'Clean/noisy'):
                trainer.inspect_data(args)
if __name__ == '__main__':
    unittest.main()
