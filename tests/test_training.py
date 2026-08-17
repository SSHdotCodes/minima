import torch

from minima.training import (
    _masked_cosine_loss,
    _masked_relative_mse_loss,
    _pooled_cosine_loss,
    _pooled_relational_loss,
)


def test_representation_losses_ignore_padding_and_match_identical_states():
    torch.manual_seed(8)
    teacher = torch.randn(2, 4, 6)
    student = teacher.clone()
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    student[mask.eq(0)] = 1000
    torch.testing.assert_close(_masked_cosine_loss(student, teacher, mask), torch.tensor(0.0))
    torch.testing.assert_close(_pooled_cosine_loss(student, teacher, mask), torch.tensor(0.0))


def test_pooled_loss_detects_mean_representation_mismatch():
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    student = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.long)
    torch.testing.assert_close(_pooled_cosine_loss(student, teacher, mask), torch.tensor(1.0))


def test_relative_mse_and_relational_losses_preserve_encoder_geometry():
    teacher = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0]],
    ])
    mask = torch.ones((2, 2), dtype=torch.long)
    assert _masked_relative_mse_loss(teacher, teacher, mask).item() == 0.0
    assert _pooled_relational_loss(teacher, teacher, mask).item() == 0.0
    scaled = teacher * 2
    assert _masked_relative_mse_loss(scaled, teacher, mask).item() > 0.0
    assert _pooled_relational_loss(scaled, teacher, mask).item() < 1.0e-6
