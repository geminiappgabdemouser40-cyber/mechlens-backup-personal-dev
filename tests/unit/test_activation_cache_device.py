"""Explicit cache placement preserves legacy computation and autograd."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache


def make_cache():
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_layers=1, n_heads=2, device="cpu", normalization_type="LN")
    )
    heads = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3).requires_grad_()
    data = {
        "blocks.0.attn.hook_result": heads,
        "blocks.0.hook_resid_post": heads.sum(dim=2) + 2,
        "hook_embed": torch.ones(1, 2, 3, dtype=torch.float64),
        "ln_final.hook_scale": torch.full((1, 2, 1), 2.0, dtype=torch.float64),
    }
    return ActivationCache(data, model, device="cpu"), heads


@pytest.mark.parametrize("explicit_none", [False, True])
def test_constructor_default_preserves_dictionary_and_tensor_identity(explicit_none):
    data = {"hook_embed": torch.ones(1, 2, 3)}
    kwargs = {"device": None} if explicit_none else {}
    with pytest.warns(FutureWarning, match="Implicit ActivationCache device") as caught:
        cache = ActivationCache(data, None, False, **kwargs)
    assert len(caught) == 1
    assert cache.cache_dict is data
    assert cache["hook_embed"] is data["hook_embed"]
    assert cache.has_batch_dim is False


def test_constructor_explicit_device_moves_all_entries_without_mutating_input():
    data = {"hook_embed": torch.ones(1, 2, 3), "other": torch.zeros(2)}
    model = SimpleNamespace(weight=torch.ones(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cache = ActivationCache(data, model, device=torch.device("meta"))
    assert all(t.device.type == "meta" for t in cache.values())
    assert all(t.device.type == "cpu" for t in data.values())
    assert cache.model is model
    assert model.weight.device.type == "cpu"


def test_empty_constructor_needs_no_first_tensor():
    data = {}
    with pytest.warns(FutureWarning):
        cache = ActivationCache(data, None)
    assert cache.cache_dict is data
    assert len(ActivationCache({}, None, device="cpu")) == 0


@pytest.mark.parametrize("method", ["stack_head_results", "stack_activation"])
def test_stack_default_warns_and_matches_explicit_cpu(method):
    cache, _ = make_cache()
    args = ("resid_post",) if method == "stack_activation" else ()
    with pytest.warns(FutureWarning, match="Implicit ActivationCache device") as caught:
        legacy = getattr(cache, method)(*args)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        explicit = getattr(cache, method)(*args, device="cpu")
    torch.testing.assert_close(legacy, explicit)


@pytest.mark.parametrize("method", ["stack_head_results", "stack_activation"])
def test_stack_explicit_destination_does_not_move_cache_or_model(method):
    cache, _ = make_cache()
    original = dict(cache.cache_dict)
    args = ("resid_post",) if method == "stack_activation" else ()
    result = getattr(cache, method)(*args, device=torch.device("meta"))
    assert result.device.type == "meta"
    assert result.dtype == torch.float64
    assert cache.model.cfg.device == "cpu"
    assert all(cache.cache_dict[key] is value for key, value in original.items())


def test_head_stack_labels_remainder_slice_and_ln_unchanged():
    cache, _ = make_cache()
    with pytest.warns(FutureWarning):
        legacy, labels = cache.stack_head_results(1, True, True, (0, 1), True)
    actual, actual_labels = cache.stack_head_results(1, True, True, (0, 1), True, device="cpu")
    torch.testing.assert_close(actual, legacy)
    assert labels == actual_labels == ["L0H0", "L0H1", "remainder"]
    assert actual.shape == (3, 1, 1, 3)


def test_empty_head_stack_honors_destination_and_return_labels():
    cache, _ = make_cache()
    result, labels = cache.stack_head_results(0, True, device="meta")
    assert result.shape == (0, 1, 2, 3)
    assert result.device.type == "meta"
    assert labels == []


def test_activation_stack_preserves_positional_arguments_and_gradients():
    cache, heads = make_cache()
    result = cache.stack_activation("result", 1, "attn", device="cpu")
    result.sum().backward()
    torch.testing.assert_close(heads.grad, torch.ones_like(heads))


def test_head_stack_preserves_gradients():
    cache, heads = make_cache()
    cache.stack_head_results(device="cpu").sum().backward()
    torch.testing.assert_close(heads.grad, torch.ones_like(heads))


@pytest.mark.parametrize("method", ["stack_head_results", "stack_activation"])
def test_stack_none_keeps_current_device_even_if_model_config_differs(method):
    cache, _ = make_cache()
    cache.model.cfg.device = "meta"
    args = ("resid_post",) if method == "stack_activation" else ()
    with pytest.warns(FutureWarning):
        result = getattr(cache, method)(*args, device=None)
    assert result.device.type == "cpu"


def test_empty_head_stack_uses_cache_device_and_honors_explicit_override():
    cache, _ = make_cache()
    cache.model.cfg.device = "meta"
    with pytest.warns(FutureWarning):
        legacy = cache.stack_head_results(0)
    explicit = cache.stack_head_results(0, device="meta")
    assert legacy.device.type == "cpu"
    assert explicit.device.type == "meta"
    assert explicit.shape == legacy.shape


@pytest.mark.parametrize("projected,apply_ln", [(False, False), (True, False), (True, True)])
def test_empty_neuron_stack_uses_cache_device(projected, apply_ln):
    cache, _ = make_cache()
    cache.model.cfg.device = "meta"
    cache.model.cfg.d_mlp = 4
    cache.cache_dict["blocks.0.ln1.hook_scale"] = torch.ones(1, 2, 1)
    projection = torch.ones(3, 2, dtype=torch.float64) if projected else None
    result, labels = cache.stack_neuron_results(
        0, return_labels=True, apply_ln=apply_ln, project_output_onto=projection
    )
    assert result.device.type == "cpu"
    assert result.shape == (0, 1, 2, 2 if projected else 3)
    assert labels == []


@pytest.mark.parametrize("method", ["head", "neuron", "projected_neuron", "ln_projected_neuron"])
@pytest.mark.parametrize("cache_device,model_device", [("cpu", "meta"), ("meta", "cpu")])
def test_empty_allocations_follow_cache_after_to(method, cache_device, model_device):
    cache, _ = make_cache()
    cache.model.cfg.device = model_device
    cache.model.cfg.d_mlp = 4
    cache.cache_dict["blocks.0.ln1.hook_scale"] = torch.ones(1, 2, 1)
    cache.to(cache_device)
    if method == "head":
        with pytest.warns(FutureWarning):
            result, labels = cache.stack_head_results(0, return_labels=True)
    else:
        projected = "projected" in method
        projection = (
            torch.ones(3, 2, dtype=torch.float64, device=cache_device) if projected else None
        )
        result, labels = cache.stack_neuron_results(
            0,
            return_labels=True,
            apply_ln=method == "ln_projected_neuron",
            project_output_onto=projection,
        )
    assert result.device.type == cache_device
    assert result.shape[:3] == (0, 1, 2)
    assert labels == []
    assert cache.model.cfg.device == model_device


def test_empty_head_stack_explicit_destination_after_layernorm():
    cache, _ = make_cache()
    cache.cache_dict["blocks.0.ln1.hook_scale"] = torch.ones(1, 2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result, labels = cache.stack_head_results(
            0, return_labels=True, apply_ln=True, device="meta"
        )
    assert result.device.type == "meta"
    assert result.shape == (0, 1, 2, 3)
    assert labels == []
    assert all(t.device.type == "cpu" for t in cache.values())


@pytest.mark.parametrize("method", ["stack_head_results", "stack_activation"])
def test_explicit_device_preserves_nonempty_values_and_labels(method):
    cache, heads = make_cache()
    if method == "stack_head_results":
        result, labels = cache.stack_head_results(return_labels=True, device=torch.device("cpu"))
        expected = heads.permute(2, 0, 1, 3)
        assert labels == ["L0H0", "L0H1"]
    else:
        result = cache.stack_activation("result", 1, "attn", device=torch.device("cpu"))
        expected = heads.unsqueeze(0)
    torch.testing.assert_close(result, expected)
    assert result.dtype == heads.dtype
