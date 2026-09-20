"""Multi-threaded ranged downloader for hf-mirror.

Single-connection throughput on this machine is ~40 KB/s, which makes a 3 GB
checkpoint a multi-hour job.  hf-mirror advertises ``Accept-Ranges: bytes``, so
we split each file into segments and pull them in parallel.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from typing import Optional

CHUNK = 1 << 18


def head(url: str) -> tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.headers.get("Accept-Ranges") == "bytes"


def _fetch_segment(url: str, start: int, end: int, fpath: str, lock: threading.Lock,
                   state: dict, idx: int, retries: int = 5) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "curl/8"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                got = 0
                want = end - start + 1
                buf = bytearray()
                while got < want:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    buf += chunk
                    got += len(chunk)
                    with lock:
                        state[idx] = got
            with lock:
                with open(fpath, "r+b") as f:
                    f.seek(start)
                    f.write(buf)
            return got == want
        except Exception as e:
            if attempt == retries - 1:
                print(f"   seg{idx} FAILED {type(e).__name__}: {e}", flush=True)
                return False
            time.sleep(2 + attempt * 2)
    return False


def download(url: str, dst: str, nthreads: int = 8, expect: int = 0) -> bool:
    total, ranged = head(url)
    if expect and total != expect:
        print(f"   size mismatch head={total} expect={expect}", flush=True)
        return False
    if not ranged or total == 0:
        nthreads = 1
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.truncate(total)

    lock = threading.Lock()

    bounds = []
    if nthreads == 1:
        bounds = [(0, total - 1)]
    else:
        seg = total // nthreads
        for i in range(nthreads):
            s = i * seg
            e = (total - 1) if i == nthreads - 1 else (s + seg - 1)
            bounds.append((s, e))

    pending = list(bounds)
    for rnd in range(4):
        if not pending:
            break
        state = {i: 0 for i in range(len(pending))}
        threads = []
        for i, (s, e) in enumerate(pending):
            t = threading.Thread(target=_fetch_segment,
                                 args=(url, s, e, tmp, lock, state, i), daemon=True)
            t.start()
            threads.append(t)

        t0 = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with lock:
                done = sum(state.values())
            el = time.time() - t0
            print(f"   round{rnd} {done/1e6:.0f}/{sum(e - s + 1 for s, e in pending)/1e6:.0f} MB"
                  f"  {done/el/1024:.0f} KB/s", flush=True)
        for t in threads:
            t.join()

        # a segment that failed after all retries leaves a hole of zeros while
        # the file size still looks correct -- re-request exactly those ranges
        want = [e - s + 1 for s, e in pending]
        failed = [(s, e) for (i, (s, e)) in enumerate(pending) if state.get(i, 0) != want[i]]
        if failed:
            print(f"   round{rnd}: {len(failed)} segment(s) incomplete, retrying", flush=True)
        pending = failed

    ok = os.path.getsize(tmp) == total and total > 0 and not pending
    if ok:
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(tmp, dst)
    else:
        print(f"   INCOMPLETE: {len(pending)} segment(s) still missing", flush=True)
    return ok


if __name__ == "__main__":
    # speed probe: 4 parallel segments of 8 MB each from the bge checkpoint
    url = "https://hf-mirror.com/BAAI/bge-base-zh-v1.5/resolve/main/pytorch_model.bin"
    total, ranged = head(url)
    print("total", total, "ranged", ranged)
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_probe.bin")
    t0 = time.time()
    N = 4
    SEG = 8 << 20
    with open(probe, "wb") as f:
        f.truncate(N * SEG)
    lock = threading.Lock()
    state = {i: 0 for i in range(N)}
    ths = []
    for i in range(N):
        ths.append(threading.Thread(target=_fetch_segment,
                                    args=(url, i * SEG, (i + 1) * SEG - 1, probe, lock, state, i),
                                    daemon=True))
        ths[i].start()
    for t in ths:
        t.join()
    el = time.time() - t0
    print(f"parallel {N}x{SEG>>20}MB in {el:.1f}s -> {N*SEG/el/1024:.0f} KB/s")
    os.remove(probe)
