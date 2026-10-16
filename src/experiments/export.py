"""Escritura de resultados y metadatos del dataset congelado (Persona 5)."""
import hashlib
import json
from pathlib import Path

from ..costs.cost_matrix import TOLL_FILENAME

NOTA_TOLLS_COMPLETE = ("tolls_complete=true solo significa que toll_eur.csv no tiene celdas vacías (NaN). "
                       "No significa que los peajes estén validados ni que cubran todos los tramos de peaje.")
METODO_HASH = ("git hash-object: SHA-1 del blob de Git, con los finales de línea CRLF convertidos a LF "
               "(el mismo valor que publica el README)")


def metadatos_dataset(data_dir):
    """Versión, hash y fecha de consulta del dataset, leídos de metadata.json."""
    meta = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    return {"version": meta["dataset_version"], "hash": meta["dataset_hash_sha256_short"],
            "query_date": meta["query_date_utc"]}


def git_hash_object(path):
    """Mismo valor que `git hash-object <path>` para los CSV y YAML de entrada.

    En el repositorio están guardados con LF; en Windows (core.autocrlf=true) la copia de trabajo tiene CRLF y
    git los vuelve a LF antes de hashear. Aquí se hace lo mismo, así el hash no depende del sistema operativo.
    No hace falta tener git instalado.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hashes_entradas(root, cfg, algo_config_paths):
    """Hash de peajes, vehículo y parámetros de los optimizadores (el hash de metadata.json no los cubre)."""
    root = Path(root)
    paths = [Path(cfg["dataset"]["directory"]) / TOLL_FILENAME, Path(cfg["vehicle_config"]),
             *map(Path, algo_config_paths)]
    files = {}
    for path in paths:
        full = root / path
        try:
            key = full.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:  # archivo fuera del repositorio (por ejemplo, en las pruebas)
            key = full.as_posix()
        files[key] = git_hash_object(full)
    return {"method": METODO_HASH, "files": files}


def guardar_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path
