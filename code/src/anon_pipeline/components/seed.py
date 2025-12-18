from __future__ import annotations

import hashlib
import hmac
from hashlib import sha256
from typing import Callable

import numpy as np

DigestFn = Callable[[bytes], "hashlib._Hash"]


class SeedGenerator:
    def generate(self, quantized: np.ndarray) -> str:
        raise NotImplementedError


class HmacSeedGenerator(SeedGenerator):
    def __init__(self, secret_key: str, digest: Callable[[bytes], "hashlib._Hash"] = sha256) -> None:
        self.secret_key = secret_key.encode("utf-8")
        self.digest = digest

    def generate(self, quantized: np.ndarray) -> str:
        payload = quantized.astype("float32").tobytes()
        digest = hmac.new(self.secret_key, payload, self.digest)
        return digest.hexdigest()
