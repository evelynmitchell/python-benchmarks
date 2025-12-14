"""
Benchmark threading operations - critical for comparing GIL vs free-threaded Python.

These benchmarks are designed to show the difference between:
- Python 3.14 (GIL enabled) - threads are serialized for CPU-bound work
- Python 3.14t (free-threaded) - true parallel execution possible
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import pytest


class TestThreadCreation:
    """Test thread creation and management overhead."""

    def test_benchmark_thread_create_join(self, benchmark):
        """Benchmark creating and joining a single thread."""

        def worker():
            pass

        def create_and_join():
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        benchmark(create_and_join)

    def test_benchmark_thread_create_10(self, benchmark):
        """Benchmark creating and joining 10 threads."""

        def worker():
            pass

        def create_many():
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        benchmark(create_many)

    def test_benchmark_thread_pool_submit(self, benchmark):
        """Benchmark ThreadPoolExecutor task submission."""

        def worker(n):
            return n * 2

        def submit_tasks():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker, i) for i in range(100)]
                return [f.result() for f in as_completed(futures)]

        result = benchmark(submit_tasks)
        assert len(result) == 100


class TestCPUBoundThreading:
    """
    Test CPU-bound work in threads - this is where free-threading shines.

    With GIL: Threads are serialized, no speedup from multiple threads
    Without GIL: True parallel execution, potential linear speedup
    """

    def test_benchmark_cpu_work_single_thread(self, benchmark):
        """Baseline: CPU-bound work in single thread."""

        def cpu_work(n):
            """Simulate CPU-bound work."""
            total = 0
            for i in range(n):
                total += i * i
            return total

        result = benchmark(cpu_work, 100000)
        assert result > 0

    def test_benchmark_cpu_work_4_threads(self, benchmark):
        """CPU-bound work split across 4 threads."""

        def cpu_work(n):
            total = 0
            for i in range(n):
                total += i * i
            return total

        def parallel_cpu():
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Split 100000 iterations across 4 threads
                futures = [executor.submit(cpu_work, 25000) for _ in range(4)]
                return sum(f.result() for f in futures)

        result = benchmark(parallel_cpu)
        assert result > 0

    def test_benchmark_cpu_work_8_threads(self, benchmark):
        """CPU-bound work split across 8 threads."""

        def cpu_work(n):
            total = 0
            for i in range(n):
                total += i * i
            return total

        def parallel_cpu():
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(cpu_work, 12500) for _ in range(8)]
                return sum(f.result() for f in futures)

        result = benchmark(parallel_cpu)
        assert result > 0


class TestSharedStateAccess:
    """
    Test concurrent access to shared state.

    With GIL: Implicit synchronization, but serialized access
    Without GIL: Needs explicit synchronization, but parallel reads possible
    """

    def test_benchmark_counter_increment_single(self, benchmark):
        """Baseline: Counter increment in single thread."""

        def increment():
            counter = 0
            for _ in range(10000):
                counter += 1
            return counter

        result = benchmark(increment)
        assert result == 10000

    def test_benchmark_counter_with_lock(self, benchmark):
        """Counter increment with lock - thread-safe pattern."""
        lock = threading.Lock()
        counter = [0]  # Mutable container

        def increment_with_lock():
            for _ in range(1000):
                with lock:
                    counter[0] += 1

        def run_threads():
            counter[0] = 0
            threads = [threading.Thread(target=increment_with_lock) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return counter[0]

        result = benchmark(run_threads)
        assert result == 4000

    def test_benchmark_queue_producer_consumer(self, benchmark):
        """Producer-consumer pattern with thread-safe queue."""

        def run_producer_consumer():
            q = Queue()
            results = []

            def producer():
                for i in range(100):
                    q.put(i)
                q.put(None)  # Sentinel

            def consumer():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    results.append(item * 2)

            prod = threading.Thread(target=producer)
            cons = threading.Thread(target=consumer)

            prod.start()
            cons.start()
            prod.join()
            cons.join()

            return len(results)

        result = benchmark(run_producer_consumer)
        assert result == 100


class TestThreadLocalStorage:
    """Test thread-local storage performance."""

    def test_benchmark_thread_local_access(self, benchmark):
        """Benchmark thread-local variable access."""
        local_data = threading.local()

        def access_local():
            local_data.value = 0
            for _ in range(1000):
                local_data.value += 1
            return local_data.value

        result = benchmark(access_local)
        assert result == 1000

    def test_benchmark_thread_local_multi_thread(self, benchmark):
        """Thread-local access from multiple threads."""
        local_data = threading.local()
        results = []

        def worker():
            local_data.value = 0
            for _ in range(1000):
                local_data.value += 1
            results.append(local_data.value)

        def run_threads():
            results.clear()
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return sum(results)

        result = benchmark(run_threads)
        assert result == 4000


class TestSynchronizationPrimitives:
    """Test performance of threading synchronization primitives."""

    def test_benchmark_lock_acquire_release(self, benchmark):
        """Benchmark lock acquire/release cycle."""
        lock = threading.Lock()

        def lock_cycle():
            for _ in range(1000):
                lock.acquire()
                lock.release()

        benchmark(lock_cycle)

    def test_benchmark_rlock_acquire_release(self, benchmark):
        """Benchmark reentrant lock acquire/release."""
        rlock = threading.RLock()

        def rlock_cycle():
            for _ in range(1000):
                rlock.acquire()
                rlock.release()

        benchmark(rlock_cycle)

    def test_benchmark_semaphore(self, benchmark):
        """Benchmark semaphore acquire/release."""
        sem = threading.Semaphore(10)

        def sem_cycle():
            for _ in range(1000):
                sem.acquire()
                sem.release()

        benchmark(sem_cycle)

    def test_benchmark_event_wait_set(self, benchmark):
        """Benchmark event set/wait cycle."""

        def event_cycle():
            event = threading.Event()
            event.set()
            for _ in range(1000):
                event.wait()
                event.clear()
                event.set()

        benchmark(event_cycle)

    def test_benchmark_condition_notify(self, benchmark):
        """Benchmark condition variable notify."""
        cond = threading.Condition()

        def condition_cycle():
            with cond:
                for _ in range(1000):
                    cond.notify()

        benchmark(condition_cycle)


class TestBarrierPerformance:
    """Test barrier synchronization for parallel algorithms."""

    def test_benchmark_barrier_4_threads(self, benchmark):
        """Benchmark barrier synchronization with 4 threads."""

        def run_with_barrier():
            barrier = threading.Barrier(4)
            results = []

            def worker(worker_id):
                for phase in range(10):
                    # Do some work
                    result = worker_id * phase
                    barrier.wait()  # Synchronize
                    results.append(result)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            return len(results)

        result = benchmark(run_with_barrier)
        assert result == 40  # 4 threads * 10 phases
