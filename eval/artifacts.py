import hashlib
from pathlib import Path


def fingerprint(path):
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix in {".bin", ".safetensors", ".json", ".txt"}
        )
    else:
        return {}
    result = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        result[file.name] = digest.hexdigest()
    return result
