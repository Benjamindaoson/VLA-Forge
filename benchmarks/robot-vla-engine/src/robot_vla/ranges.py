"""Seekable HTTP access with bounded cache, explicit byte ranges and retry validation."""

import http.client
import io
import time
import urllib.request
from collections import OrderedDict


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=512 * 1024, cache_blocks=64, retries=4):
        self.url, self.size, self.block_size = url, size, block_size
        self.cache_blocks, self.retries = cache_blocks, retries
        self.position = 0
        self.cache = OrderedDict()
        self.bytes_fetched = 0
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid whence")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        error = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{self.size}"
                    ):
                        raise OSError("server did not honor exact byte range")
                    payload = response.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise OSError("incomplete HTTP range payload")
                self.bytes_fetched += len(payload)
                self.requests += 1
                self.cache[index] = payload
                while len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
                return payload
            except (OSError, ValueError, http.client.IncompleteRead) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(attempt + 1, 3))
        raise OSError(
            f"range read failed at bytes {start}-{end}: {type(error).__name__}"
        ) from error

    def read(self, size=-1):
        if self.closed:
            raise ValueError("closed reader")
        size = min(self.size - self.position, size) if size >= 0 else self.size - self.position
        if size <= 0:
            return b""
        result = bytearray()
        while size:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            length = min(size, len(block) - offset)
            result.extend(block[offset : offset + length])
            self.position += length
            size -= length
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
