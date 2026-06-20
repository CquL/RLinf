import torch

from rlinf.algorithms.advantages import compute_grpo_advantages


def test_grpo_advantages_default_loss_mask_for_embodied_rollouts():
    rewards = torch.tensor([[1.0, 3.0], [5.0, 1.0]])

    advantages, returns = compute_grpo_advantages(
        rewards=rewards,
        group_size=2,
        n_steps=3,
        loss_mask=None,
    )

    expected_group_advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True) + 1e-6
    )
    expected = expected_group_advantages.reshape(1, -1).expand(3, -1)

    assert returns is None
    assert advantages.shape == (3, 4)
    torch.testing.assert_close(advantages, expected)
