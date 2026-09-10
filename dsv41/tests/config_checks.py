# SPDX-License-Identifier: Apache-2.0
"""Exercise the EngramConfig half of the patch.

The kernel tests build `ParallelEngramEmbedding` directly and never construct
an `EngramConfig`, so nothing there executes `vllm/config/engram.py`. This
does: the new fields exist with the documented defaults, and `compute_hash`
separates a disk-backed run from a resident one without hashing the directory.
"""

import sys

import conftest  # noqa: F401  registers the PR modules under vllm.*

from vllm.config.engram import EngramConfig  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"{'PASS' if ok else 'FAIL'} {name}{' ' + detail if detail else ''}")
    failures += not ok


default = EngramConfig()
check(
    "defaults are unchanged for resident use",
    (default.cpu_offload, default.table_path) == (True, None),
    f"(cpu_offload={default.cpu_offload}, table_path={default.table_path})",
)
check(
    "disk knobs default to 12 threads and O_DIRECT",
    (default.disk_read_threads, default.disk_direct_io) == (12, True),
)

disk_a = EngramConfig(table_path="/mnt/nvme/a")
disk_b = EngramConfig(table_path="/mnt/nvme/b")
check(
    "the table directory does not change the hash",
    disk_a.compute_hash() == disk_b.compute_hash(),
)
check(
    "disk-backed and resident hash differently",
    disk_a.compute_hash() != default.compute_hash(),
)
check(
    "read threads change the hash",
    EngramConfig(table_path="/mnt/nvme/a", disk_read_threads=4).compute_hash()
    != disk_a.compute_hash(),
)

print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
sys.exit(1 if failures else 0)
