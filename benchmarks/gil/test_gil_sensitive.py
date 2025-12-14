"""
Benchmark GIL-sensitive operations.

These benchmarks specifically target operations where the GIL has significant impact:
- Reference counting overhead
- Object allocation during multi-threading
- Mixed I/O and CPU workloads
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# Optional numpy import for numerical benchmarks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestReferenceCounting:
    """
    Test reference counting overhead.

    In free-threaded Python, reference counting uses atomic operations
    which may have different performance characteristics.
    """

    def test_benchmark_refcount_single_object(self, benchmark):
        """Benchmark reference count changes on single object."""
        obj = [1, 2, 3, 4, 5]

        def refcount_changes():
            refs = []
            for _ in range(1000):
                refs.append(obj)  # Increment refcount
            refs.clear()  # Decrement refcounts
            return sys.getrefcount(obj)

        benchmark(refcount_changes)

    def test_benchmark_refcount_many_objects(self, benchmark):
        """Benchmark creating many objects (many refcount inits)."""

        def create_objects():
            objects = []
            for i in range(1000):
                obj = {"id": i, "data": list(range(10))}
                objects.append(obj)
            return len(objects)

        result = benchmark(create_objects)
        assert result == 1000

    def test_benchmark_shared_object_threads(self, benchmark):
        """Multiple threads accessing shared object (refcount contention)."""
        shared_list = list(range(1000))

        def access_shared():
            local_ref = shared_list  # Increment refcount
            total = sum(local_ref)  # Access data
            del local_ref  # Decrement refcount
            return total

        def run_threads():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(access_shared) for _ in range(100)]
                return sum(f.result() for f in futures)

        result = benchmark(run_threads)
        assert result > 0


class TestObjectAllocationContention:
    """
    Test object allocation under thread contention.

    Memory allocator behavior differs between GIL and free-threaded builds.
    """

    def test_benchmark_concurrent_list_creation(self, benchmark):
        """Multiple threads creating lists concurrently."""

        def create_lists():
            return [list(range(100)) for _ in range(100)]

        def run_threads():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_lists) for _ in range(4)]
                return sum(len(f.result()) for f in futures)

        result = benchmark(run_threads)
        assert result == 400

    def test_benchmark_concurrent_dict_creation(self, benchmark):
        """Multiple threads creating dicts concurrently."""

        def create_dicts():
            return [{i: i * 2 for i in range(100)} for _ in range(100)]

        def run_threads():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_dicts) for _ in range(4)]
                return sum(len(f.result()) for f in futures)

        result = benchmark(run_threads)
        assert result == 400

    def test_benchmark_concurrent_string_creation(self, benchmark):
        """Multiple threads creating strings concurrently."""

        def create_strings():
            return [f"string_{i}" * 10 for i in range(1000)]

        def run_threads():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_strings) for _ in range(4)]
                return sum(len(f.result()) for f in futures)

        result = benchmark(run_threads)
        assert result == 4000


class TestMixedIOCPU:
    """
    Test mixed I/O and CPU workloads.

    GIL is released during I/O, so this tests transition overhead.
    """

    def test_benchmark_io_then_cpu(self, benchmark):
        """Simulated I/O followed by CPU work."""

        def io_then_cpu():
            # Simulate I/O (GIL released)
            time.sleep(0.0001)
            # CPU work (GIL held)
            total = sum(i * i for i in range(1000))
            return total

        benchmark(io_then_cpu)

    def test_benchmark_interleaved_io_cpu(self, benchmark):
        """Interleaved I/O and CPU work."""

        def interleaved():
            total = 0
            for _ in range(10):
                time.sleep(0.00001)  # Brief I/O
                total += sum(range(100))  # Brief CPU
            return total

        benchmark(interleaved)

    def test_benchmark_parallel_io_cpu_mix(self, benchmark):
        """Parallel threads with mixed I/O and CPU."""

        def mixed_work(worker_id):
            total = 0
            for i in range(10):
                if i % 2 == 0:
                    time.sleep(0.0001)  # I/O
                else:
                    total += sum(range(1000))  # CPU
            return total

        def run_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(mixed_work, i) for i in range(4)]
                return sum(f.result() for f in futures)

        result = benchmark(run_parallel)
        assert result > 0


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestNumpyThreading:
    """
    Test NumPy operations under threading.

    NumPy releases the GIL for many operations, making it a good
    test case for mixed GIL/non-GIL code.
    """

    def test_benchmark_numpy_single_thread(self, benchmark):
        """Baseline: NumPy operations single-threaded."""

        def numpy_work():
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            return np.dot(a, b).sum()

        result = benchmark(numpy_work)
        assert result != 0

    def test_benchmark_numpy_parallel_independent(self, benchmark):
        """Parallel NumPy operations on independent data."""

        def numpy_work():
            a = np.random.rand(500, 500)
            b = np.random.rand(500, 500)
            return np.dot(a, b).sum()

        def run_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(numpy_work) for _ in range(4)]
                return sum(f.result() for f in futures)

        result = benchmark(run_parallel)
        assert result != 0

    def test_benchmark_numpy_shared_array_read(self, benchmark):
        """Multiple threads reading from shared NumPy array."""
        shared_array = np.random.rand(1000, 1000)

        def read_array():
            return shared_array.sum()

        def run_parallel():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(read_array) for _ in range(100)]
                return sum(f.result() for f in futures)

        result = benchmark(run_parallel)
        assert result != 0


class TestGILReleasePoints:
    """
    Test performance around GIL release points.

    Certain operations release the GIL; this tests the overhead
    of acquiring/releasing the GIL.
    """

    def test_benchmark_sleep_zero(self, benchmark):
        """Benchmark sleep(0) - forces GIL release."""

        def sleep_many():
            for _ in range(100):
                time.sleep(0)

        benchmark(sleep_many)

    def test_benchmark_frequent_gil_release(self, benchmark):
        """Frequent GIL release/acquire cycles."""

        def frequent_release():
            total = 0
            for i in range(100):
                time.sleep(0)  # Release GIL
                total += i * i  # Hold GIL
            return total

        result = benchmark(frequent_release)
        assert result > 0

    def test_benchmark_file_io_gil_release(self, benchmark):
        """File I/O (releases GIL)."""
        import io

        def file_io():
            buf = io.BytesIO()
            for _ in range(100):
                buf.write(b"x" * 1000)
            buf.seek(0)
            return len(buf.read())

        result = benchmark(file_io)
        assert result == 100000


class TestAtomicOperations:
    """
    Test operations that need atomicity.

    In free-threaded Python, these may use different synchronization.
    """

    def test_benchmark_dict_concurrent_update(self, benchmark):
        """Concurrent dict updates (needs internal synchronization)."""
        lock = threading.Lock()

        def update_dict():
            d = {}
            threads = []

            def updater(start):
                for i in range(start, start + 250):
                    with lock:
                        d[i] = i * 2

            for i in range(4):
                t = threading.Thread(target=updater, args=(i * 250,))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            return len(d)

        result = benchmark(update_dict)
        assert result == 1000

    def test_benchmark_list_concurrent_append(self, benchmark):
        """Concurrent list appends (needs synchronization)."""
        lock = threading.Lock()

        def append_to_list():
            lst = []
            threads = []

            def appender():
                for i in range(250):
                    with lock:
                        lst.append(i)

            for _ in range(4):
                t = threading.Thread(target=appender)
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            return len(lst)

        result = benchmark(append_to_list)
        assert result == 1000
