import hashlib
import io

import pandas as pd
import pytest

from reviews import data


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def remote_for(payload: bytes, size: int | None = None) -> data.RemoteFile:
    return data.RemoteFile(
        "amazon_polarity/x.parquet",
        len(payload) if size is None else size,
        hashlib.sha256(payload).hexdigest(),
    )


def test_fetch_downloads_and_checks_hash(tmp_path):
    payload = b"parquet bytes"
    urls = []

    def opener(url):
        urls.append(url)
        return FakeResponse(payload)

    path = data.fetch(remote_for(payload), tmp_path, opener)
    assert path.read_bytes() == payload
    assert data.REVISION in urls[0]
    # A second call uses the file on disk.
    data.fetch(remote_for(payload), tmp_path, lambda url: pytest.fail("downloaded again"))


def test_fetch_rejects_wrong_hash_and_deletes_file(tmp_path):
    remote = remote_for(b"expected")
    with pytest.raises(ValueError, match="does not match"):
        data.fetch(remote, tmp_path, lambda url: FakeResponse(b"swapped!"))
    assert not list(tmp_path.iterdir())


def test_fetch_rejects_a_changed_file_already_on_disk(tmp_path):
    payload = b"expected"
    (tmp_path / "x.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        data.fetch(remote_for(payload), tmp_path, lambda url: FakeResponse(payload))


def test_fetch_stops_a_download_larger_than_pinned(tmp_path):
    remote = remote_for(b"small", size=5)
    with pytest.raises(ValueError, match="larger than"):
        data.fetch(remote, tmp_path, lambda url: FakeResponse(b"x" * (3 << 20)))
    assert not list(tmp_path.iterdir())


def test_read_rejects_unknown_labels_and_fills_missing_text(tmp_path):
    path = tmp_path / "a.parquet"
    pd.DataFrame({"label": [0, 1], "title": [None, "t"], "content": ["c", None]}).to_parquet(path)
    df = data.read(path)
    assert df["title"].tolist() == ["", "t"] and df["content"].tolist() == ["c", ""]

    pd.DataFrame({"label": [0, 2], "title": ["a", "b"], "content": ["c", "d"]}).to_parquet(path)
    with pytest.raises(ValueError, match="labels"):
        data.read(path)


@pytest.mark.parametrize(
    "a, b",
    [
        (("Great!", "Works  fine."), ("great", "works fine")),
        (("Café", "nice"), ("cafe", "NICE")),
        (("", "same text"), ("same", "text")),
    ],
)
def test_dedupe_key_ignores_case_accents_punctuation_and_spacing(a, b):
    assert data.dedupe_key(*a) == data.dedupe_key(*b)


def test_dedupe_key_keeps_different_words_apart():
    assert data.dedupe_key("good", "not bad") != data.dedupe_key("good", "bad")
    assert data.dedupe_key("", "") == data.dedupe_key("", "  ...  ")


def frame(texts, labels):
    return data.add_keys(pd.DataFrame({"label": labels, "title": "", "content": texts}))


def test_split_drops_reviews_that_are_in_test_and_repeats():
    pool = frame(["a", "b", "c", "d", "e", "E!", "leak"], [0, 1, 0, 1, 0, 0, 1])
    test = frame(["Leak.", "x", "x"], [1, 0, 1])
    train, validation, report = data.split_and_dedupe(pool, test, validation_size=2, seed=0)

    kept = set(train["key"]) | set(validation["key"])
    assert not kept & set(test["key"])
    assert not set(train["key"]) & set(validation["key"])
    assert len(train) + len(validation) == 5
    assert report["pool_in_test"] == 1 and report["pool_repeats"] == 1
    assert report["test_repeats"] == 1 and report["conflicting_labels_in_test"] == 1
    assert len(test) == 3


def test_split_is_reproducible():
    pool = frame([str(i) for i in range(50)], [i % 2 for i in range(50)])
    test = frame(["t"], [0])
    first = data.split_and_dedupe(pool, test, 10, seed=3)
    second = data.split_and_dedupe(pool, test, 10, seed=3)
    pd.testing.assert_frame_equal(first[0], second[0])


def test_stratified_sample_keeps_class_and_shard_shares():
    shard_a = pd.DataFrame({"label": [0] * 600 + [1] * 400, "title": "", "content": "a"})
    shard_b = pd.DataFrame({"label": [0] * 500 + [1] * 1500, "title": "", "content": "b"})
    sample = data.stratified_sample([shard_a, shard_b], n=300, seed=0)

    assert len(sample) == 300
    counts = sample.groupby(["content", "label"]).size().to_dict()
    assert counts == {("a", 0): 60, ("a", 1): 40, ("b", 0): 50, ("b", 1): 150}


def test_stratified_sample_never_repeats_a_row():
    shard = pd.DataFrame(
        {"label": [0, 1] * 50, "title": "", "content": [str(i) for i in range(100)]}
    )
    sample = data.stratified_sample([shard], n=100, seed=1)
    assert sample["content"].is_unique


def test_validation_has_the_same_class_share_as_the_pool():
    labels = [1] * 700 + [0] * 300
    pool = frame([str(i) for i in range(1000)], labels)
    train, validation, _ = data.split_and_dedupe(pool, frame(["t"], [0]), 100, seed=0)
    assert validation["label"].mean() == 0.7 and train["label"].mean() == 0.7
