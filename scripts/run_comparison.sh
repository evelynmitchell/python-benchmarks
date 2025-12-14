#!/usr/bin/env bash
#
# Run benchmarks across multiple Python versions and compare results.
#
# Usage:
#   ./scripts/run_comparison.sh [category]
#
# Examples:
#   ./scripts/run_comparison.sh              # Run all benchmarks
#   ./scripts/run_comparison.sh threading    # Run only threading benchmarks
#   ./scripts/run_comparison.sh interpreter  # Run only interpreter benchmarks

set -e

CATEGORY="${1:-}"
RESULTS_DIR="benchmark_results/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Python versions to test
VERSIONS=("3.12" "3.14" "3.14t")

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

echo "=============================================="
echo "Python Benchmark Comparison"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Versions: ${VERSIONS[*]}"
echo ""

# Set up virtual environments and run benchmarks
for VERSION in "${VERSIONS[@]}"; do
    VENV_DIR=".venv${VERSION//./}"

    echo "----------------------------------------------"
    echo "Setting up Python $VERSION..."
    echo "----------------------------------------------"

    # Install Python version if needed
    uv python install "$VERSION" 2>/dev/null || true

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        uv venv --python "$VERSION" "$VENV_DIR"
        uv pip install pytest pytest-benchmark pytest-asyncio numpy --python "$VENV_DIR/bin/python"
    fi

    # Check GIL status
    GIL_STATUS=$("$VENV_DIR/bin/python" -c "import sys; print('enabled' if sys._is_gil_enabled() else 'disabled')")
    echo "Python $VERSION - GIL: $GIL_STATUS"

    # Determine which benchmarks to run
    if [ -n "$CATEGORY" ]; then
        BENCHMARK_PATH="benchmarks/$CATEGORY/"
    else
        BENCHMARK_PATH="benchmarks/"
    fi

    # Run benchmarks
    echo "Running benchmarks..."
    "$VENV_DIR/bin/pytest" "$BENCHMARK_PATH" \
        --benchmark-only \
        --benchmark-json="$RESULTS_DIR/results_$VERSION.json" \
        --benchmark-min-rounds=5 \
        --ignore=benchmarks/pytorch/ \
        -q 2>&1 | tee "$RESULTS_DIR/output_$VERSION.txt"

    echo ""
done

# Generate comparison
echo "=============================================="
echo "Generating comparison..."
echo "=============================================="

python3 << EOF
import json
import glob
from pathlib import Path

results_dir = "$RESULTS_DIR"
results = {}

for f in glob.glob(f"{results_dir}/results_*.json"):
    version = Path(f).stem.replace("results_", "")
    with open(f) as fp:
        data = json.load(fp)
        results[version] = {
            b["name"]: {
                "mean": b["stats"]["mean"],
                "stddev": b["stats"]["stddev"],
                "min": b["stats"]["min"],
            }
            for b in data.get("benchmarks", [])
        }

# Find common benchmarks
if results:
    common = set.intersection(*[set(r.keys()) for r in results.values()])
    print(f"\nCompared {len(common)} common benchmarks across {len(results)} versions\n")

    # Compare GIL vs free-threaded if both exist
    if "3.14" in results and "3.14t" in results:
        print("=" * 60)
        print("GIL vs Free-threaded (3.14 vs 3.14t)")
        print("=" * 60)

        comparisons = []
        for name in common:
            gil = results["3.14"][name]["mean"]
            ft = results["3.14t"][name]["mean"]
            ratio = gil / ft if ft > 0 else 1.0
            comparisons.append((name, ratio, gil * 1000, ft * 1000))

        # Sort by ratio (biggest wins for free-threading first)
        comparisons.sort(key=lambda x: -x[1])

        print("\nTop 10 - Free-threading wins:")
        for name, ratio, gil_ms, ft_ms in comparisons[:10]:
            short_name = name.split("::")[-1][:45]
            print(f"  {ratio:5.2f}x  {short_name:<45} ({gil_ms:.3f}ms vs {ft_ms:.3f}ms)")

        print("\nTop 10 - GIL wins:")
        for name, ratio, gil_ms, ft_ms in comparisons[-10:]:
            short_name = name.split("::")[-1][:45]
            print(f"  {ratio:5.2f}x  {short_name:<45} ({gil_ms:.3f}ms vs {ft_ms:.3f}ms)")

        # Summary statistics
        faster_ft = sum(1 for _, r, _, _ in comparisons if r > 1.1)
        slower_ft = sum(1 for _, r, _, _ in comparisons if r < 0.9)
        neutral = len(comparisons) - faster_ft - slower_ft

        print(f"\nSummary:")
        print(f"  Free-threaded faster (>10%): {faster_ft}")
        print(f"  GIL faster (>10%):           {slower_ft}")
        print(f"  Similar (<10% difference):   {neutral}")

# Save comparison report
with open(f"{results_dir}/comparison_report.txt", "w") as f:
    f.write("Benchmark Comparison Report\n")
    f.write("=" * 60 + "\n")
    for version in sorted(results.keys()):
        f.write(f"\nPython {version}: {len(results[version])} benchmarks\n")

print(f"\nResults saved to: {results_dir}/")
EOF

echo ""
echo "Done! Results in: $RESULTS_DIR"
