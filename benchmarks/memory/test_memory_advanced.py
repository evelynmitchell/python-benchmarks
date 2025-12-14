"""
Advanced memory profiling benchmarks.

These tests combine time benchmarking with memory profiling to understand
both speed and memory characteristics of different approaches.
"""

import gc
import sys
import tracemalloc

import pytest


class TestMemoryVsSpeed:
    """Test memory/speed tradeoffs in common patterns."""

    def test_benchmark_list_vs_generator_memory(self, benchmark):
        """
        Compare list (eager, memory-heavy) vs generator (lazy, memory-light).

        List: Allocates all items upfront
        Generator: Produces items on demand
        """

        def list_approach():
            data = [x * 2 for x in range(100000)]
            return sum(data)

        result = benchmark(list_approach)
        assert result > 0

    def test_benchmark_generator_approach(self, benchmark):
        """Generator approach - lower memory, same result."""

        def generator_approach():
            return sum(x * 2 for x in range(100000))

        result = benchmark(generator_approach)
        assert result > 0

    def test_benchmark_slots_memory_savings(self, benchmark):
        """
        Compare __slots__ vs regular class memory usage.

        __slots__ avoids per-instance __dict__, saving memory.
        """

        class RegularClass:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        def create_regular():
            return [RegularClass(i, i * 2, i * 3) for i in range(1000)]

        result = benchmark(create_regular)
        assert len(result) == 1000

    def test_benchmark_slots_class(self, benchmark):
        """__slots__ class - should be faster and use less memory."""

        class SlotsClass:
            __slots__ = ["x", "y", "z"]

            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        def create_slots():
            return [SlotsClass(i, i * 2, i * 3) for i in range(1000)]

        result = benchmark(create_slots)
        assert len(result) == 1000


class TestDataStructureMemory:
    """Compare memory usage of different data structures."""

    def test_benchmark_tuple_vs_list(self, benchmark):
        """Tuple (immutable) vs List (mutable) - tuples are smaller."""

        def create_tuples():
            return [tuple(range(10)) for _ in range(10000)]

        result = benchmark(create_tuples)
        assert len(result) == 10000

    def test_benchmark_list_creation(self, benchmark):
        """List creation - compare with tuple."""

        def create_lists():
            return [list(range(10)) for _ in range(10000)]

        result = benchmark(create_lists)
        assert len(result) == 10000

    def test_benchmark_namedtuple_vs_dict(self, benchmark):
        """NamedTuple vs dict - namedtuple is more memory efficient."""
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y", "z"])

        def create_namedtuples():
            return [Point(i, i * 2, i * 3) for i in range(10000)]

        result = benchmark(create_namedtuples)
        assert len(result) == 10000

    def test_benchmark_dict_creation(self, benchmark):
        """Dict creation - compare with namedtuple."""

        def create_dicts():
            return [{"x": i, "y": i * 2, "z": i * 3} for i in range(10000)]

        result = benchmark(create_dicts)
        assert len(result) == 10000

    def test_benchmark_dataclass_creation(self, benchmark):
        """Dataclass creation - modern alternative."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int
            z: int

        def create_dataclasses():
            return [Point(i, i * 2, i * 3) for i in range(10000)]

        result = benchmark(create_dataclasses)
        assert len(result) == 10000

    def test_benchmark_dataclass_slots(self, benchmark):
        """Dataclass with slots - best of both worlds."""
        from dataclasses import dataclass

        @dataclass(slots=True)
        class Point:
            x: int
            y: int
            z: int

        def create_dataclasses_slots():
            return [Point(i, i * 2, i * 3) for i in range(10000)]

        result = benchmark(create_dataclasses_slots)
        assert len(result) == 10000


class TestMemoryProfiling:
    """Tests that include memory profiling information."""

    @pytest.fixture
    def memory_tracker(self):
        """Fixture to track memory during tests."""
        gc.collect()
        tracemalloc.start()
        yield
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n  Memory: current={current/1024:.1f}KB, peak={peak/1024:.1f}KB")

    def test_benchmark_string_concat_memory(self, benchmark, memory_tracker):
        """String concatenation - creates many intermediate strings."""

        def string_concat():
            s = ""
            for i in range(1000):
                s += str(i)
            return s

        result = benchmark(string_concat)
        assert len(result) > 0

    def test_benchmark_string_join_memory(self, benchmark, memory_tracker):
        """String join - single allocation."""

        def string_join():
            return "".join(str(i) for i in range(1000))

        result = benchmark(string_join)
        assert len(result) > 0

    def test_benchmark_list_append_memory(self, benchmark, memory_tracker):
        """List append - reallocates as it grows."""

        def list_append():
            lst = []
            for i in range(10000):
                lst.append(i)
            return lst

        result = benchmark(list_append)
        assert len(result) == 10000

    def test_benchmark_list_preallocated_memory(self, benchmark, memory_tracker):
        """Preallocated list - no reallocation needed."""

        def list_preallocated():
            lst = [None] * 10000
            for i in range(10000):
                lst[i] = i
            return lst

        result = benchmark(list_preallocated)
        assert len(result) == 10000


class TestObjectSize:
    """Benchmark operations that create objects of different sizes."""

    def test_benchmark_small_objects(self, benchmark):
        """Many small objects - tests allocator efficiency."""

        def create_small():
            return [i for i in range(100000)]

        result = benchmark(create_small)
        assert len(result) == 100000

    def test_benchmark_medium_objects(self, benchmark):
        """Fewer medium objects."""

        def create_medium():
            return [list(range(100)) for _ in range(1000)]

        result = benchmark(create_medium)
        assert len(result) == 1000

    def test_benchmark_large_objects(self, benchmark):
        """Few large objects."""

        def create_large():
            return [list(range(10000)) for _ in range(10)]

        result = benchmark(create_large)
        assert len(result) == 10


class TestGCImpact:
    """Test garbage collection impact on performance."""

    def test_benchmark_with_gc_enabled(self, benchmark):
        """Normal operation with GC enabled."""

        def with_gc():
            data = []
            for i in range(1000):
                obj = {"id": i, "data": list(range(100))}
                data.append(obj)
                if i % 100 == 0:
                    data = data[-10:]  # Keep only last 10
            return len(data)

        result = benchmark(with_gc)
        assert result == 10

    def test_benchmark_with_gc_disabled(self, benchmark):
        """Same operation with GC temporarily disabled."""

        def without_gc():
            gc.disable()
            try:
                data = []
                for i in range(1000):
                    obj = {"id": i, "data": list(range(100))}
                    data.append(obj)
                    if i % 100 == 0:
                        data = data[-10:]
                return len(data)
            finally:
                gc.enable()
                gc.collect()

        result = benchmark(without_gc)
        assert result == 10

    def test_benchmark_gc_collect_frequency(self, benchmark):
        """Test impact of explicit gc.collect() calls."""

        def frequent_gc():
            total = 0
            for i in range(100):
                data = [list(range(100)) for _ in range(100)]
                total += sum(len(d) for d in data)
                gc.collect()  # Explicit collection
            return total

        result = benchmark(frequent_gc)
        assert result > 0


class TestRefCountBehavior:
    """Test reference counting behavior and its performance impact."""

    def test_benchmark_simple_references(self, benchmark):
        """Simple reference creation/deletion."""

        def simple_refs():
            obj = [1, 2, 3, 4, 5]
            refs = []
            for _ in range(1000):
                refs.append(obj)  # Increment refcount
            refs.clear()  # Decrement refcounts
            return sys.getrefcount(obj)

        result = benchmark(simple_refs)
        assert result >= 2

    def test_benchmark_circular_references(self, benchmark):
        """Circular reference creation (requires GC to clean)."""

        def circular_refs():
            objects = []
            for _ in range(100):
                a = {}
                b = {}
                a["ref"] = b
                b["ref"] = a
                objects.append(a)
            objects.clear()
            return gc.collect()

        result = benchmark(circular_refs)
        assert result >= 0

    def test_benchmark_weakref_usage(self, benchmark):
        """Weak references - no refcount overhead."""
        import weakref

        class MyClass:
            pass

        def use_weakrefs():
            objects = [MyClass() for _ in range(1000)]
            refs = [weakref.ref(obj) for obj in objects]
            # Check how many are still alive
            alive = sum(1 for r in refs if r() is not None)
            return alive

        result = benchmark(use_weakrefs)
        assert result == 1000


class TestInternedStrings:
    """Test string interning behavior."""

    def test_benchmark_short_strings(self, benchmark):
        """Short strings - may be interned automatically."""

        def short_strings():
            strings = []
            for i in range(1000):
                s = f"s{i}"  # Short strings
                strings.append(s)
            return len(strings)

        result = benchmark(short_strings)
        assert result == 1000

    def test_benchmark_long_strings(self, benchmark):
        """Long strings - not interned."""

        def long_strings():
            strings = []
            for i in range(1000):
                s = f"this_is_a_much_longer_string_{i}" * 10
                strings.append(s)
            return len(strings)

        result = benchmark(long_strings)
        assert result == 1000

    def test_benchmark_explicit_intern(self, benchmark):
        """Explicitly interned strings."""

        def interned_strings():
            strings = []
            for i in range(1000):
                s = sys.intern(f"interned_{i}")
                strings.append(s)
            return len(strings)

        result = benchmark(interned_strings)
        assert result == 1000
