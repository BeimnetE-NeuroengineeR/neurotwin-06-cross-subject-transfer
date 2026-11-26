import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def parse_line(line: str) -> dict:
    match = re.search(
        r"Shift_Scale=([0-9.]+)\s+ZeroShot_Acc=([0-9.]+)\s+Adapted_Acc=([0-9.]+)\s+Transfer_Gain=([+-]?[0-9.]+)",
        line,
    )
    if not match:
        raise ValueError(f"Could not parse transfer summary line: {line}")
    return {
        "Shift_Scale": float(match.group(1)),
        "ZeroShot_Acc": float(match.group(2)),
        "Adapted_Acc": float(match.group(3)),
        "Transfer_Gain": float(match.group(4)),
    }


def main():
    print("Running Project 6: C++17 Cross-Subject Transfer Shift Sweep...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpp_src = os.path.join(repo_root, "src/cross_subject_transfer.cpp")
    cpp_bin = os.path.join(repo_root, "src/cross_subject_transfer")
    results_dir = os.path.join(repo_root, "results")
    figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    subprocess.run(["clang++", "-std=c++17", "-O2", cpp_src, "-o", cpp_bin], check=True)

    shift_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    rows = []

    for sc in shift_scales:
        proc = subprocess.run(
            [cpp_bin, f"{sc:.2f}", "320", "0.55"],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(parse_line(proc.stdout.strip()))

    df = pd.DataFrame(rows)
    sweep_csv = os.path.join(results_dir, "shift_sweep.csv")
    df.to_csv(sweep_csv, index=False)

    eval_csv = os.path.join(results_dir, "eval_shift1.00.csv")
    subprocess.run(
        [cpp_bin, "1.00", "320", "0.55", eval_csv],
        capture_output=True,
        text=True,
        check=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    ax0 = axes[0]
    ax0.plot(df["Shift_Scale"], df["ZeroShot_Acc"], marker="o", lw=1.8, label="Zero-Shot Decoder")
    ax0.plot(df["Shift_Scale"], df["Adapted_Acc"], marker="s", lw=1.8, linestyle="--", label="Adapted Decoder")
    ax0.set_title("Decoding Accuracy Across Domain Shift Scale")
    ax0.set_xlabel("Shift Scale")
    ax0.set_ylabel("Accuracy")
    ax0.set_ylim(0.0, 1.05)
    ax0.legend(loc="best")
    ax0.grid(True, linestyle=":", alpha=0.5)

    ax1 = axes[1]
    ax1.bar(df["Shift_Scale"].astype(str), df["Transfer_Gain"], width=0.5)
    ax1.axhline(0.0, lw=1.0, color="black")
    ax1.set_title("Transfer Gain Across Domain Shift Scales")
    ax1.set_xlabel("Shift Scale")
    ax1.set_ylabel("Transfer Gain (Adapted - Zero-Shot)")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_png = os.path.join(figures_dir, "main_result.png")
    plt.savefig(out_png)

    print("\nCross-Subject Transfer Sweep Results:")
    print(df.to_string(index=False))
    print("\nSaved sweep to results/shift_sweep.csv and plot to figures/main_result.png!")


if __name__ == "__main__":
    main()
