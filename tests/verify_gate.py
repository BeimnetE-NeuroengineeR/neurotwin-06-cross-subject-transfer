import csv
import math
import os
import re
import subprocess


def test_cross_subject_transfer_gate():
    print("Running Project 6 Sanity Check: C++17 Cross-Subject Transfer...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpp_src = os.path.join(repo_root, "src/cross_subject_transfer.cpp")
    cpp_bin = os.path.join(repo_root, "src/cross_subject_transfer")
    out_csv = os.path.join(repo_root, "results/gate_transfer.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    assert os.path.isfile(cpp_src), f"Missing C++ source at {cpp_src}"

    subprocess.run(
        ["clang++", "-std=c++17", "-O2", cpp_src, "-o", cpp_bin],
        check=True,
    )

    proc = subprocess.run(
        [cpp_bin, "1.00", "320", "0.55", out_csv],
        capture_output=True,
        text=True,
        check=True,
    )
    line = proc.stdout.strip()
    print(line)

    match = re.search(
        r"Shift_Scale=([0-9.]+)\s+ZeroShot_Acc=([0-9.]+)\s+Adapted_Acc=([0-9.]+)\s+Transfer_Gain=([+-]?[0-9.]+)",
        line,
    )
    assert match is not None, f"Could not parse transfer summary line: {line}"

    zs_acc = float(match.group(2))
    ad_acc = float(match.group(3))
    gain = float(match.group(4))

    assert 0.0 <= zs_acc <= 1.0, f"ZeroShot_Acc out of bounds: {zs_acc}"
    assert 0.0 <= ad_acc <= 1.0, f"Adapted_Acc out of bounds: {ad_acc}"
    assert math.isfinite(gain) and abs(gain - (ad_acc - zs_acc)) < 1e-3, "Transfer_Gain mismatch"

    assert os.path.isfile(out_csv), "Transfer evaluation CSV was not created"
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 240, f"Expected 240 test rows, got {len(rows)}"
    for r in rows:
        assert int(r["ZeroShot_Correct"]) in (0, 1)
        assert int(r["Adapted_Correct"]) in (0, 1)

    os.remove(out_csv)
    print("Cross-subject transfer gate passed! All 240 test trials valid.")


if __name__ == "__main__":
    test_cross_subject_transfer_gate()
