"""Save a trained model with a manifest, and only load it if its hash still matches.

A .joblib file is a pickle, and unpickling runs code, so the hash is checked before loading
(decision 6). A hash passed in by the caller beats the manifest's own copy, because someone who
can swap the model file can usually edit the manifest next to it too.
"""

import hashlib
import hmac
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

MODEL_FILE, MANIFEST_FILE = "model.joblib", "model.json"


class ModelIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    model: object
    manifest: dict


def save_model(model, out_dir: Path, info: dict | None = None) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    data = buffer.getvalue()
    (out_dir / MODEL_FILE).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "sha256": digest,
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **(info or {}),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return digest


def load_model(model_dir: Path, expected_sha256: str | None = None) -> LoadedModel:
    try:
        manifest = json.loads((model_dir / MANIFEST_FILE).read_text())
        expected = str(expected_sha256 or manifest["sha256"]).lower()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelIntegrityError("model manifest missing or unreadable") from exc
    try:
        data = (model_dir / MODEL_FILE).read_bytes()
    except OSError as exc:
        raise ModelIntegrityError("model file missing") from exc
    # Unpickle exactly the bytes that were hashed; hashing and then reopening the file would
    # leave a gap in which it could be swapped.
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected):
        raise ModelIntegrityError("model file hash does not match; refusing to load it")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ModelIntegrityError(
            f"model saved with scikit-learn {manifest.get('sklearn_version')}, "
            f"running {sklearn.__version__}; retrain instead of loading it"
        )
    try:
        model = joblib.load(io.BytesIO(data))
    except Exception as exc:
        # The bytes are the ones that were saved, so this is a code change (a class that moved or
        # was renamed), not tampering. Unpickling can raise almost anything, hence the broad catch.
        raise ModelIntegrityError(
            f"model passed the hash check but can't be loaded: {exc}"
        ) from exc
    return LoadedModel(model, manifest)
