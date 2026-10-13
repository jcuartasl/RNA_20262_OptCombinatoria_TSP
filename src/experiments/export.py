"""Escritura de resultados y metadatos del dataset congelado (Persona 5)."""
import json
from pathlib import Path


def metadatos_dataset(data_dir):
    """Versión, hash y fecha de consulta del dataset, leídos de metadata.json."""
    meta = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    return {"version": meta["dataset_version"], "hash": meta["dataset_hash_sha256_short"],
            "query_date": meta["query_date_utc"]}


def guardar_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path
