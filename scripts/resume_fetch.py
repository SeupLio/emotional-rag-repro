"""Stall-resistant, resumable downloader for hf-mirror.

The previous threaded ranged downloader pre-allocates the file, so a segment
that stalls leaves a hole of zeros while the size check still passes -- two
checkpoints were silently corrupted before we noticed.  Worse, connections here
regularly stall indefinitely (the server trickles instead of closing, so socket
timeouts never fire).

This version:

* splits the file into small chunks (default 8 MB);
* writes bytes **progressively**, so an abandoned chunk keeps what it got;
* abandons a chunk after ``stall_timeout`` seconds without progress and pushes
  the *remaining* sub-range back onto the queue;
* records completed ranges in ``<dst>.ranges.json`` so a restart resumes
  instead of starting over.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from typing import Optional

READ = 1 << 18


def head(url: str) -> tuple[int, bool]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.headers.get("Accept-Ranges") == "bytes"


def _load_ranges(meta: str) -> list[list[int]]:
    if os.path.exists(meta):
        try:
            return json.loads(open(meta, encoding="utf-8").read())
        except Exception:
            return []
    return []


def _save_ranges(meta: str, rs) -> None:
    tmp = meta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted(rs), f)
    os.replace(tmp, meta)


def download(url: str, dst: str, nthreads: int = 8, expect: int = 0,
             chunk: int = 8 << 20, stall_timeout: float = 25.0,
             max_seconds: float = 3600.0) -> bool:
    total, _ = head(url)
    if expect and total != expect:
        print(f"   size mismatch head={total} expect={expect}", flush=True)
        return False
    if total <= 0:
        return False

    meta = dst + ".ranges.json"
    part = dst + ".part"
    if os.path.exists(dst) and not os.path.exists(part):
        return True
    if not os.path.exists(part):
        with open(part, "wb") as f:
            f.truncate(total)

    done = [tuple(r) for r in _load_ranges(meta)]
    done_set = set(done)
    pending: "queue.Queue" = queue.Queue()
    for s in range(0, total, chunk):
        e = min(total - 1, s + chunk - 1)
        if (s, e) not in done_set:
            pending.put((s, e))

    lock = threading.Lock()
    progress = {"bytes": sum(e - s + 1 for s, e in done)}
    t0 = time.time()

    def worker() -> None:
        f = open(part, "r+b")
        try:
            while True:
                if time.time() - t0 > max_seconds:
                    return
                try:
                    s, e = pending.get_nowait()
                except queue.Empty:
                    return
                got = 0
                want = e - s + 1
                last = time.time()
                try:
                    req = urllib.request.Request(
                        url, headers={"Range": f"bytes={s+got}-{e}", "User-Agent": "curl/8"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        while got < want:
                            buf = r.read(READ)
                            if not buf:
                                break
                            with lock:
                                f.seek(s + got)
                                f.write(buf)
                            got += len(buf)
                            progress["bytes"] += len(buf)
                            last = time.time()
                            if time.time() - t0 > max_seconds:
                                break
                        if got == want:
                            with lock:
                                done_set.add((s, e))
                            continue
                except Exception:
                    pass
                if time.time() - last > stall_timeout or got < want:
                    # keep whatever we managed to write and requeue the rest
                    if got > 0:
                        with lock:
                            done_set.add((s, s + got - 1))
                    if s + got <= e:
                        pending.put((s + got, e))
        finally:
            f.close()

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(nthreads)]
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(5)
        with lock:
            p = progress["bytes"]
            _save_ranges(meta, [list(r) for r in done_set])
        print(f"   {p/1e6:.0f}/{total/1e6:.0f} MB  {p/(time.time()-t0)/1024:.0f} KB/s", flush=True)
    for t in ths:
        t.join()

    # final sweep: retry whatever is still missing a few more times
    for rnd in range(3):
        if pending.empty():
            break
        left = []
        while not pending.empty():
            left.append(pending.get())
        print(f"   round{rnd}: {len(left)} chunk(s) left", flush=True)
        for (s, e) in left:
            pending.put((s, e))
        ths = [threading.Thread(target=worker, daemon=True) for _ in range(min(nthreads, len(left)))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

    _save_ranges(meta, [list(r) for r in done_set])
    complete = pending.empty()
    if complete:
        os.replace(part, dst)
        if os.path.exists(meta):
            os.replace(meta, meta + ".done")
        print(f"   [ok] {os.path.basename(dst)} {total/1e6:.1f} MB", flush=True)
    else:
        print(f"   [incomplete] {pending.qsize()} chunk(s) missing", flush=True)
    return complete


if __name__ == "__main__":
    import sys
    url, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    download(url, dst, nthreads=n)
