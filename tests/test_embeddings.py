import numpy as np
import pytest
import torch

from reviews import embeddings


def test_mean_pool_ignores_padding_and_normalises():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = embeddings.mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0]]))


def test_mean_pool_of_an_all_padding_row_is_finite():
    pooled = embeddings.mean_pool(torch.ones(1, 2, 3), torch.zeros(1, 2, dtype=torch.long))
    assert torch.isfinite(pooled).all()


class FakeTokenizer:
    """Encodes a text as its length, so the output can be traced back to the input."""

    def __call__(self, texts, **kwargs):
        longest = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), longest)
        mask = torch.zeros(len(texts), longest, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, : len(t)] = len(t)
            mask[i, : len(t)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeModel:
    def __call__(self, input_ids, attention_mask):
        hidden = torch.zeros(*input_ids.shape, 384)
        hidden[..., 0] = input_ids
        hidden[..., 1] = 1.0
        return type("Out", (), {"last_hidden_state": hidden})()


def test_encode_keeps_input_order_despite_length_sorting():
    texts = ["a" * 9, "a", "a" * 4, "", "a" * 2]
    out = embeddings.encode(texts, FakeTokenizer(), FakeModel(), batch=2)
    assert out.shape == (5, 384) and out.dtype == np.float32
    # Before normalising, component 0 is the length and component 1 is 1, so their ratio
    # identifies which text each row came from. The empty text is encoded as one space.
    ratio = out[:, 0] / out[:, 1]
    assert np.allclose(ratio, [9, 1, 4, 1, 2])


def test_encode_of_nothing():
    assert embeddings.encode([], FakeTokenizer(), FakeModel()).shape == (0, 384)


def test_cache_is_reused_and_rebuilt_when_the_size_changes(tmp_path):
    texts = ["one", "three"]
    calls = []

    def counting_model():
        model = FakeModel()

        def run(**kw):
            calls.append(1)
            return model(**kw)

        return run

    encoder = (FakeTokenizer(), counting_model())
    first = embeddings.cached("val", texts, tmp_path, "f" * 64, encoder)
    again = embeddings.cached("val", texts, tmp_path, "f" * 64, encoder)
    assert np.array_equal(first, again) and len(calls) == 1
    bigger = embeddings.cached("val", texts + ["x"], tmp_path, "f" * 64, encoder)
    assert bigger.shape[0] == 3 and len(calls) == 2


def test_cache_file_is_not_a_pickle(tmp_path):
    embeddings.cached("val", ["x"], tmp_path, "0" * 64, (FakeTokenizer(), FakeModel()))
    (path,) = tmp_path.glob("*.npy")
    np.load(path, allow_pickle=False)
    with pytest.raises(ValueError):
        np.save(tmp_path / "bad.npy", np.array([object()]), allow_pickle=False)
