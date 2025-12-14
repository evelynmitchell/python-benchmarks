"""Examples demonstrating different pytest-benchmark modes and configurations."""

import time


class TestBenchmarkModes:
    """Demonstrate different benchmarking approaches."""

    def test_simple_benchmark(self, benchmark):
        """
        Standard benchmark - automatic calibration.

        pytest-benchmark will:
        1. Calibrate iterations to fit timer resolution
        2. Run warmup if enabled
        3. Execute multiple rounds
        4. Calculate statistics
        """
        result = benchmark(sorted, list(range(1000, 0, -1)))
        assert result == list(range(1, 1001))

    def test_pedantic_benchmark(self, benchmark):
        """
        Pedantic mode - explicit control over iterations and rounds.

        Use when you need:
        - Exact number of iterations
        - Exact number of rounds
        - No automatic calibration
        - Reproducible measurements
        """
        def sort_list():
            return sorted(list(range(1000, 0, -1)))

        result = benchmark.pedantic(
            sort_list,
            iterations=10,   # Run 10 times per round
            rounds=100,      # 100 independent measurements
            warmup_rounds=5  # 5 warmup rounds before measuring
        )
        assert result == list(range(1, 1001))

    def test_benchmark_with_setup(self, benchmark):
        """
        Benchmark with setup function - setup time excluded from measurement.

        Use when:
        - You need to prepare data before each round
        - Setup cost should not be measured
        """
        def setup():
            """Create a fresh list for each round."""
            return (list(range(10000, 0, -1)),), {}

        def sort_list(data):
            return sorted(data)

        benchmark.pedantic(
            sort_list,
            setup=setup,
            rounds=50,
            iterations=1  # One iteration per round since setup is expensive
        )


class TestStatisticsInterpretation:
    """Examples showing how to interpret benchmark statistics."""

    def test_consistent_operation(self, benchmark):
        """
        Consistent operation - expect low StdDev and IQR.

        Simple arithmetic should have:
        - Low standard deviation
        - Narrow IQR
        - Few outliers
        """
        def simple_math():
            return sum(range(1000))

        benchmark(simple_math)

    def test_variable_operation(self, benchmark):
        """
        Variable operation - may have higher StdDev.

        Operations involving memory allocation or GC may show:
        - Higher standard deviation
        - Wider IQR
        - More outliers (GC pauses)
        """
        def allocate_and_discard():
            data = [list(range(100)) for _ in range(100)]
            del data

        benchmark(allocate_and_discard)

    def test_io_bound_operation(self, benchmark):
        """
        I/O-bound operation - typically higher variance.

        Sleep simulates I/O - expect:
        - Min close to sleep time
        - Higher max due to scheduling
        """
        def simulated_io():
            time.sleep(0.0001)  # 100 microseconds

        benchmark(simulated_io)


class TestBenchmarkComparison:
    """Examples for comparing implementations."""

    def test_list_comprehension(self, benchmark):
        """Benchmark list comprehension."""
        benchmark(lambda: [x * 2 for x in range(10000)])

    def test_map_function(self, benchmark):
        """Benchmark map() - compare with list comprehension."""
        benchmark(lambda: list(map(lambda x: x * 2, range(10000))))

    def test_generator_consumed(self, benchmark):
        """Benchmark generator expression consumed immediately."""
        benchmark(lambda: sum(x * 2 for x in range(10000)))
