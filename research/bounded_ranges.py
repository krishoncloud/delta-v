"""Seekable HTTP range reader with a strict download budget."""
import io

import httpx


class BoundedRanges(io.RawIOBase):
    def __init__(self, url, size, budget=96 * 1024 * 1024):
        self.url = url
        self.size = size
        self.budget = budget
        self.position = 0
        self.cache = {}
        self.downloaded = 0
        self.client = httpx.Client(follow_redirects=True, timeout=90)
        self.block_size = 256 * 1024

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative offset")
        return self.position

    def read(self, count=-1):
        if count < 0:
            raise ValueError("Unbounded reads are forbidden")
        end = min(self.size, self.position + count)
        chunks = []
        while self.position < end:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                stop = min(self.size, start + self.block_size) - 1
                expected = stop - start + 1
                if self.downloaded + expected > self.budget:
                    raise ValueError("Source byte budget exceeded")
                with self.client.stream(
                    "GET",
                    self.url,
                    headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"},
                ) as response:
                    if response.status_code != 206:
                        raise ValueError("Server did not honor exact range; refusing full-file transfer")
                    if not response.headers.get("content-range", "").startswith(f"bytes {start}-{stop}/"):
                        raise ValueError("Unexpected content range")
                    data = response.read()
                    if len(data) != expected:
                        raise ValueError("Unexpected range length")
                self.cache[index] = data
                self.downloaded += len(data)
            offset = self.position % self.block_size
            take = min(end - self.position, len(self.cache[index]) - offset)
            chunks.append(self.cache[index][offset : offset + take])
            self.position += take
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        self.client.close()
        super().close()
