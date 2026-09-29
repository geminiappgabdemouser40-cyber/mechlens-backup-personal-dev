"""Public cache lookup, mapping views, and bounded representations."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache, ActivationCacheKey


@pytest.fixture
def cache():
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=2))
    return ActivationCache(
        {
            "hook_embed": torch.ones(1),
            "blocks.0.hook_resid_pre": torch.zeros(1),
            "blocks.1.attn.hook_q": torch.full((1,), 2.0),
        },
        model,
        device="cpu",
    )


@pytest.mark.parametrize(
    "key,canonical",
    [
        ("hook_embed", "hook_embed"),
        ("embed", "hook_embed"),
        (("embed",), "hook_embed"),
        (("embed", None), "hook_embed"),
        (("embed", None, None), "hook_embed"),
        (("resid_pre", 0), "blocks.0.hook_resid_pre"),
        (("q", 1, "attn"), "blocks.1.attn.hook_q"),
        (("q", -1, "attn"), "blocks.1.attn.hook_q"),
        (("q", -1), "blocks.1.attn.hook_q"),
    ],
)
def test_lookup_preserves_shorthand_and_negative_layers(
    cache, key: ActivationCacheKey, canonical: str
):
    assert cache[key] is cache.cache_dict[canonical]


def test_lookup_missing_key_raises_key_error(cache):
    with pytest.raises(KeyError):
        cache["missing"]


def test_mapping_views_remain_live(cache):
    keys, values, items = cache.keys(), cache.values(), cache.items()
    tensor = torch.full((1,), 7.0)
    cache.cache_dict["new"] = tensor
    assert "new" in keys
    assert any(value is tensor for value in values)
    assert any(key == "new" and value is tensor for key, value in items)
    assert list(cache) == list(keys)


def test_repr_empty_and_small(cache):
    assert repr(ActivationCache({}, None, device="cpu")) == "ActivationCache with 0 keys: []"
    text = repr(cache)
    assert "3 keys" in text
    assert "hook_embed" in text
    assert "more" not in text


def test_repr_bounds_keys_and_key_lengths_without_rendering_tensors():
    class NoReprTensor(torch.Tensor):
        def __repr__(self) -> str:
            raise AssertionError("Cache repr must not format tensors")

    tensor = torch.ones(1).as_subclass(NoReprTensor)
    data = {f"{index}-" + "x" * 10000: tensor for index in range(1000)}
    cache = ActivationCache(data, None, device="cpu")
    text = repr(cache)
    assert "1000 keys" in text
    assert "(+992 more)" in text
    assert len(text) < 800


def test_public_methods_have_argument_and_return_annotations():
    source = Path(__file__).parents[2] / "transformer_lens" / "ActivationCache.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActivationCache"
    )
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        assert node.returns is not None, node.name
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.arg != "self":
                assert arg.annotation is not None, (node.name, arg.arg)


def test_stack_activation_docstring_matches_parameters():
    assert "incl_remainder" not in ActivationCache.stack_activation.__doc__
