"""Download, verify and read the single-source MeshRIR archive."""

import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


ARCHIVE = Path("data/raw/S1-M3969_npy.zip")
URL = "https://zenodo.org/records/10852693/files/S1-M3969_npy.zip"
RECORD = "https://zenodo.org/records/10852693"
MD5 = "2cb598eb44bb9905560c545db7af3432"
SIZE_BYTES = 995777273
SAMPLE_RATE = 48000
IR_LENGTH = 32768


def verify_archive(path):
    if path.stat().st_size != SIZE_BYTES:
        raise ValueError(f"{path} has an unexpected file size")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False))
    if digest.hexdigest() != MD5:
        raise ValueError(f"{path} does not match the Zenodo checksum")


def download_archive(path):
    if path.exists():
        verify_archive(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".zip.part")
    if not partial.exists() or partial.stat().st_size < SIZE_BYTES:
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
             "--output", str(partial), URL],
            check=True,
        )
    verify_archive(partial)
    partial.replace(path)


class MeshRIR:
    """Read individual measurements directly from the ZIP."""

    def __init__(self, archive):
        self.archive = Path(archive)
        with zipfile.ZipFile(self.archive) as source:
            names = source.namelist()
            microphone_files = [
                name for name in names if PurePosixPath(name).name == "pos_mic.npy"
            ]
            if len(microphone_files) != 1:
                raise ValueError("Expected one microphone coordinate file")
            microphone_file = microphone_files[0]
            parent = PurePosixPath(microphone_file).parent
            metadata = json.loads(source.read(str(parent / "data.json")))
            if metadata["samplerate"] != SAMPLE_RATE or metadata["ir length"] != IR_LENGTH:
                raise ValueError("Unexpected sample rate or response length in MeshRIR metadata")
            self.receivers = self._read_array(source, microphone_file)
            self.sources = self._read_array(source, str(parent / "pos_src.npy"))
            self._impulses = {}
            for name in names:
                member = PurePosixPath(name)
                match = re.fullmatch(r"ir_(\d+)\.npy", member.name)
                if member.parent == parent and match:
                    index = int(match[1])
                    if index in self._impulses:
                        raise ValueError(f"Duplicate impulse response index {index}")
                    self._impulses[index] = name
        if self.receivers.shape != (3969, 3) or self.sources.shape != (1, 3):
            raise ValueError("Unexpected coordinate shapes for MeshRIR S1-M3969")
        if self._impulses.keys() != set(range(len(self.receivers))):
            raise ValueError("Impulse response indices do not match receiver coordinates")

    @staticmethod
    def _read_array(source, member):
        return np.load(io.BytesIO(source.read(member)), allow_pickle=False)

    def load(self, receiver_ids):
        with zipfile.ZipFile(self.archive) as source:
            impulses = [
                self._read_array(source, self._impulses[index]) for index in receiver_ids
            ]
        if any(ir.shape != (1, IR_LENGTH) or not np.isfinite(ir).all() for ir in impulses):
            raise ValueError("Expected finite single-source responses of length 32768")
        return np.concatenate(impulses, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_archive(args.archive)
    else:
        download_archive(args.archive)
    dataset = MeshRIR(args.archive)
    dataset.load([0])
    print(f"Verified {args.archive} with {len(dataset.receivers)} positions at {SAMPLE_RATE} Hz")
