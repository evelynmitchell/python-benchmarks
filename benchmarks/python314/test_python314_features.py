"""
Benchmark Python 3.14 specific features and optimizations.

Python 3.14 includes:
- Free-threading support (3.14t)
- Improved JIT compilation (experimental)
- Enhanced error messages
- Performance optimizations
"""

import asyncio
import sys

import pytest


def get_python_version():
    """Get Python version tuple."""
    return sys.version_info[:2]


class TestPatternMatching:
    """
    Test structural pattern matching performance (PEP 634).

    Introduced in Python 3.10, but optimizations continue.
    """

    def test_benchmark_match_literal(self, benchmark):
        """Benchmark match with literal patterns."""

        def match_literals():
            total = 0
            for i in range(1000):
                value = i % 5
                match value:
                    case 0:
                        total += 1
                    case 1:
                        total += 2
                    case 2:
                        total += 3
                    case 3:
                        total += 4
                    case 4:
                        total += 5
            return total

        result = benchmark(match_literals)
        assert result > 0

    def test_benchmark_match_vs_if_elif(self, benchmark):
        """Benchmark match statement (compare with if/elif)."""

        def with_match():
            total = 0
            for i in range(1000):
                value = i % 5
                match value:
                    case 0:
                        total += 1
                    case 1:
                        total += 2
                    case 2:
                        total += 3
                    case _:
                        total += 4
            return total

        result = benchmark(with_match)
        assert result > 0

    def test_benchmark_if_elif_baseline(self, benchmark):
        """Baseline: if/elif chain (compare with match)."""

        def with_if_elif():
            total = 0
            for i in range(1000):
                value = i % 5
                if value == 0:
                    total += 1
                elif value == 1:
                    total += 2
                elif value == 2:
                    total += 3
                else:
                    total += 4
            return total

        result = benchmark(with_if_elif)
        assert result > 0

    def test_benchmark_match_sequence(self, benchmark):
        """Benchmark match with sequence patterns."""

        def match_sequences():
            data = [
                [1, 2],
                [1, 2, 3],
                [1, 2, 3, 4],
                [],
                [1],
            ]
            total = 0
            for _ in range(200):
                for item in data:
                    match item:
                        case [x, y]:
                            total += x + y
                        case [x, y, z]:
                            total += x + y + z
                        case [x, y, z, w]:
                            total += x + y + z + w
                        case []:
                            total += 0
                        case [x]:
                            total += x
            return total

        result = benchmark(match_sequences)
        assert result > 0

    def test_benchmark_match_mapping(self, benchmark):
        """Benchmark match with mapping patterns."""

        def match_mappings():
            data = [
                {"type": "point", "x": 1, "y": 2},
                {"type": "circle", "x": 0, "y": 0, "r": 5},
                {"type": "rect", "x": 0, "y": 0, "w": 10, "h": 20},
            ]
            total = 0
            for _ in range(200):
                for item in data:
                    match item:
                        case {"type": "point", "x": x, "y": y}:
                            total += x + y
                        case {"type": "circle", "r": r}:
                            total += r
                        case {"type": "rect", "w": w, "h": h}:
                            total += w * h
            return total

        result = benchmark(match_mappings)
        assert result > 0


class TestTaskGroup:
    """
    Test asyncio.TaskGroup performance (Python 3.11+).

    TaskGroup provides structured concurrency with automatic cleanup.
    """

    def test_benchmark_taskgroup_10_tasks(self, benchmark):
        """Benchmark TaskGroup with 10 concurrent tasks."""

        async def worker(n):
            await asyncio.sleep(0)
            return n * 2

        async def run_taskgroup():
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(worker(i)) for i in range(10)]
            return [t.result() for t in tasks]

        def run():
            return asyncio.run(run_taskgroup())

        result = benchmark(run)
        assert len(result) == 10

    def test_benchmark_taskgroup_100_tasks(self, benchmark):
        """Benchmark TaskGroup with 100 concurrent tasks."""

        async def worker(n):
            await asyncio.sleep(0)
            return n * 2

        async def run_taskgroup():
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(worker(i)) for i in range(100)]
            return [t.result() for t in tasks]

        def run():
            return asyncio.run(run_taskgroup())

        result = benchmark(run)
        assert len(result) == 100

    def test_benchmark_gather_baseline(self, benchmark):
        """Baseline: asyncio.gather (compare with TaskGroup)."""

        async def worker(n):
            await asyncio.sleep(0)
            return n * 2

        async def run_gather():
            return await asyncio.gather(*[worker(i) for i in range(100)])

        def run():
            return asyncio.run(run_gather())

        result = benchmark(run)
        assert len(result) == 100


class TestExceptionGroups:
    """
    Test ExceptionGroup performance (Python 3.11+).

    ExceptionGroups allow multiple exceptions to be raised together.
    """

    def test_benchmark_exceptiongroup_create(self, benchmark):
        """Benchmark creating ExceptionGroups."""

        def create_groups():
            groups = []
            for i in range(100):
                eg = ExceptionGroup(
                    f"group_{i}",
                    [ValueError(f"error_{j}") for j in range(5)],
                )
                groups.append(eg)
            return len(groups)

        result = benchmark(create_groups)
        assert result == 100

    def test_benchmark_exceptiongroup_catch(self, benchmark):
        """Benchmark catching ExceptionGroups with except*."""

        def catch_groups():
            count = 0
            for i in range(100):
                try:
                    raise ExceptionGroup(
                        "test",
                        [ValueError("v"), TypeError("t"), RuntimeError("r")],
                    )
                except* ValueError:
                    count += 1
                except* TypeError:
                    count += 1
                except* RuntimeError:
                    count += 1
            return count

        result = benchmark(catch_groups)
        assert result == 300  # 3 exception types * 100 iterations


class TestUnionTypes:
    """Test union type performance (Python 3.10+)."""

    def test_benchmark_isinstance_union(self, benchmark):
        """Benchmark isinstance() with union types."""

        def check_types():
            values = [1, "hello", 3.14, None, True, [], {}, (1, 2)]
            count = 0
            for _ in range(1000):
                for v in values:
                    if isinstance(v, int | str | float):
                        count += 1
            return count

        result = benchmark(check_types)
        assert result > 0

    def test_benchmark_isinstance_tuple_baseline(self, benchmark):
        """Baseline: isinstance() with tuple (compare with union)."""

        def check_types():
            values = [1, "hello", 3.14, None, True, [], {}, (1, 2)]
            count = 0
            for _ in range(1000):
                for v in values:
                    if isinstance(v, (int, str, float)):
                        count += 1
            return count

        result = benchmark(check_types)
        assert result > 0


class TestWalrus:
    """Test walrus operator performance (Python 3.8+)."""

    def test_benchmark_with_walrus(self, benchmark):
        """Benchmark with walrus operator."""

        def with_walrus():
            data = list(range(1000))
            results = []
            for item in data:
                if (doubled := item * 2) > 100:
                    results.append(doubled)
            return len(results)

        result = benchmark(with_walrus)
        assert result > 0

    def test_benchmark_without_walrus(self, benchmark):
        """Baseline: without walrus operator."""

        def without_walrus():
            data = list(range(1000))
            results = []
            for item in data:
                doubled = item * 2
                if doubled > 100:
                    results.append(doubled)
            return len(results)

        result = benchmark(without_walrus)
        assert result > 0


class TestFreeThreadingIndicators:
    """
    Tests specifically designed to show free-threading benefits.

    These should show significant differences between 3.14 and 3.14t.
    """

    def test_benchmark_gil_status(self, benchmark):
        """Report GIL status (informational)."""

        def check_gil():
            return sys._is_gil_enabled()

        result = benchmark(check_gil)
        # Result will be True (3.14) or False (3.14t)
        print(f"\nGIL enabled: {result}")

    def test_benchmark_parallel_computation_indicator(self, benchmark):
        """
        Parallel computation that benefits from no-GIL.

        In 3.14t, threads can truly run in parallel.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        def cpu_work():
            total = 0
            for i in range(50000):
                total += i * i
            return total

        def parallel_work():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cpu_work) for _ in range(4)]
                return sum(f.result() for f in futures)

        result = benchmark(parallel_work)
        assert result > 0

        # Print info for comparison
        gil_status = "enabled" if sys._is_gil_enabled() else "disabled"
        print(f"\n[GIL {gil_status}] Parallel computation complete")
