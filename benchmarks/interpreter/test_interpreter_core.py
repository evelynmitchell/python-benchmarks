"""
Benchmark core Python interpreter operations.

These benchmarks measure fundamental interpreter performance:
- Function call overhead
- Attribute access
- Dictionary operations
- Exception handling
- Iteration patterns
"""

import pytest


class TestFunctionCalls:
    """Test function call overhead - fundamental to all Python code."""

    def test_benchmark_empty_function_call(self, benchmark):
        """Benchmark calling an empty function - pure call overhead."""

        def empty():
            pass

        def call_many():
            for _ in range(10000):
                empty()

        benchmark(call_many)

    def test_benchmark_function_with_args(self, benchmark):
        """Benchmark function call with positional arguments."""

        def with_args(a, b, c):
            return a + b + c

        def call_many():
            total = 0
            for i in range(10000):
                total += with_args(i, i + 1, i + 2)
            return total

        result = benchmark(call_many)
        assert result > 0

    def test_benchmark_function_with_kwargs(self, benchmark):
        """Benchmark function call with keyword arguments."""

        def with_kwargs(a=0, b=0, c=0):
            return a + b + c

        def call_many():
            total = 0
            for i in range(10000):
                total += with_kwargs(a=i, b=i + 1, c=i + 2)
            return total

        result = benchmark(call_many)
        assert result > 0

    def test_benchmark_function_with_star_args(self, benchmark):
        """Benchmark function call with *args/**kwargs."""

        def with_star(*args, **kwargs):
            return sum(args) + sum(kwargs.values())

        def call_many():
            total = 0
            for i in range(10000):
                total += with_star(i, i + 1, x=i + 2, y=i + 3)
            return total

        result = benchmark(call_many)
        assert result > 0

    def test_benchmark_nested_function_calls(self, benchmark):
        """Benchmark nested function calls."""

        def level3(x):
            return x + 1

        def level2(x):
            return level3(x) + 1

        def level1(x):
            return level2(x) + 1

        def call_nested():
            total = 0
            for i in range(10000):
                total += level1(i)
            return total

        result = benchmark(call_nested)
        assert result > 0

    def test_benchmark_recursive_function(self, benchmark):
        """Benchmark recursive function calls."""

        def recursive_sum(n):
            if n <= 0:
                return 0
            return n + recursive_sum(n - 1)

        def call_recursive():
            total = 0
            for _ in range(100):
                total += recursive_sum(100)
            return total

        result = benchmark(call_recursive)
        assert result > 0

    def test_benchmark_closure_call(self, benchmark):
        """Benchmark calling a closure."""

        def make_adder(n):
            def adder(x):
                return x + n

            return adder

        add_10 = make_adder(10)

        def call_closure():
            total = 0
            for i in range(10000):
                total += add_10(i)
            return total

        result = benchmark(call_closure)
        assert result > 0

    def test_benchmark_lambda_call(self, benchmark):
        """Benchmark lambda function calls."""
        add = lambda x, y: x + y  # noqa: E731

        def call_lambda():
            total = 0
            for i in range(10000):
                total += add(i, i + 1)
            return total

        result = benchmark(call_lambda)
        assert result > 0


class TestAttributeAccess:
    """Test attribute access patterns."""

    def test_benchmark_instance_attribute_read(self, benchmark):
        """Benchmark reading instance attributes."""

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2
                self.z = 3

        obj = Obj()

        def read_attrs():
            total = 0
            for _ in range(10000):
                total += obj.x + obj.y + obj.z
            return total

        result = benchmark(read_attrs)
        assert result > 0

    def test_benchmark_instance_attribute_write(self, benchmark):
        """Benchmark writing instance attributes."""

        class Obj:
            def __init__(self):
                self.x = 0

        obj = Obj()

        def write_attrs():
            for i in range(10000):
                obj.x = i

        benchmark(write_attrs)

    def test_benchmark_slots_attribute_read(self, benchmark):
        """Benchmark reading __slots__ attributes (optimized)."""

        class SlotsObj:
            __slots__ = ["x", "y", "z"]

            def __init__(self):
                self.x = 1
                self.y = 2
                self.z = 3

        obj = SlotsObj()

        def read_slots():
            total = 0
            for _ in range(10000):
                total += obj.x + obj.y + obj.z
            return total

        result = benchmark(read_slots)
        assert result > 0

    def test_benchmark_property_access(self, benchmark):
        """Benchmark property descriptor access."""

        class PropObj:
            def __init__(self):
                self._value = 42

            @property
            def value(self):
                return self._value

        obj = PropObj()

        def read_property():
            total = 0
            for _ in range(10000):
                total += obj.value
            return total

        result = benchmark(read_property)
        assert result > 0

    def test_benchmark_getattr_dynamic(self, benchmark):
        """Benchmark dynamic attribute access with getattr()."""

        class Obj:
            def __init__(self):
                self.attr1 = 1
                self.attr2 = 2
                self.attr3 = 3

        obj = Obj()
        attrs = ["attr1", "attr2", "attr3"]

        def dynamic_access():
            total = 0
            for _ in range(10000):
                for attr in attrs:
                    total += getattr(obj, attr)
            return total

        result = benchmark(dynamic_access)
        assert result > 0


class TestDictionaryOperations:
    """Test dictionary operation performance."""

    def test_benchmark_dict_get(self, benchmark):
        """Benchmark dict key lookup."""
        d = {i: i * 2 for i in range(1000)}

        def dict_gets():
            total = 0
            for i in range(1000):
                total += d[i]
            return total

        result = benchmark(dict_gets)
        assert result > 0

    def test_benchmark_dict_get_method(self, benchmark):
        """Benchmark dict.get() method."""
        d = {i: i * 2 for i in range(1000)}

        def dict_get_method():
            total = 0
            for i in range(1000):
                total += d.get(i, 0)
            return total

        result = benchmark(dict_get_method)
        assert result > 0

    def test_benchmark_dict_set(self, benchmark):
        """Benchmark dict key assignment."""

        def dict_sets():
            d = {}
            for i in range(1000):
                d[i] = i * 2
            return len(d)

        result = benchmark(dict_sets)
        assert result == 1000

    def test_benchmark_dict_update(self, benchmark):
        """Benchmark dict.update() method."""
        base = {i: i for i in range(500)}
        update_data = {i: i * 2 for i in range(500, 1000)}

        def dict_update():
            d = base.copy()
            d.update(update_data)
            return len(d)

        result = benchmark(dict_update)
        assert result == 1000

    def test_benchmark_dict_in_check(self, benchmark):
        """Benchmark 'in' membership test."""
        d = {i: i * 2 for i in range(1000)}

        def in_checks():
            count = 0
            for i in range(2000):
                if i in d:
                    count += 1
            return count

        result = benchmark(in_checks)
        assert result == 1000

    def test_benchmark_dict_iteration(self, benchmark):
        """Benchmark dict iteration."""
        d = {i: i * 2 for i in range(1000)}

        def iterate_dict():
            total = 0
            for k, v in d.items():
                total += k + v
            return total

        result = benchmark(iterate_dict)
        assert result > 0


class TestListOperations:
    """Test list operation performance."""

    def test_benchmark_list_index(self, benchmark):
        """Benchmark list indexing."""
        lst = list(range(1000))

        def list_index():
            total = 0
            for i in range(1000):
                total += lst[i]
            return total

        result = benchmark(list_index)
        assert result > 0

    def test_benchmark_list_append(self, benchmark):
        """Benchmark list.append()."""

        def list_appends():
            lst = []
            for i in range(1000):
                lst.append(i)
            return len(lst)

        result = benchmark(list_appends)
        assert result == 1000

    def test_benchmark_list_extend(self, benchmark):
        """Benchmark list.extend()."""
        data = list(range(100))

        def list_extends():
            lst = []
            for _ in range(10):
                lst.extend(data)
            return len(lst)

        result = benchmark(list_extends)
        assert result == 1000

    def test_benchmark_list_comprehension(self, benchmark):
        """Benchmark list comprehension."""

        def list_comp():
            return [i * 2 for i in range(1000)]

        result = benchmark(list_comp)
        assert len(result) == 1000

    def test_benchmark_list_slice(self, benchmark):
        """Benchmark list slicing."""
        lst = list(range(1000))

        def list_slices():
            total = 0
            for i in range(0, 900, 100):
                total += sum(lst[i : i + 100])
            return total

        result = benchmark(list_slices)
        assert result > 0


class TestExceptionHandling:
    """Test exception handling overhead."""

    def test_benchmark_try_no_exception(self, benchmark):
        """Benchmark try/except when no exception raised."""

        def no_exception():
            total = 0
            for i in range(1000):
                try:
                    total += i
                except ValueError:
                    pass
            return total

        result = benchmark(no_exception)
        assert result > 0

    def test_benchmark_try_with_exception(self, benchmark):
        """Benchmark try/except when exception is raised and caught."""

        def with_exception():
            count = 0
            for i in range(1000):
                try:
                    if i % 10 == 0:
                        raise ValueError("test")
                    count += 1
                except ValueError:
                    pass
            return count

        result = benchmark(with_exception)
        assert result == 900

    def test_benchmark_exception_chain(self, benchmark):
        """Benchmark exception chaining (raise from)."""

        def exception_chain():
            count = 0
            for i in range(100):
                try:
                    try:
                        raise ValueError("inner")
                    except ValueError as e:
                        raise RuntimeError("outer") from e
                except RuntimeError:
                    count += 1
            return count

        result = benchmark(exception_chain)
        assert result == 100


class TestIterationPatterns:
    """Test various iteration patterns."""

    def test_benchmark_for_range(self, benchmark):
        """Benchmark for loop with range()."""

        def for_range():
            total = 0
            for i in range(10000):
                total += i
            return total

        result = benchmark(for_range)
        assert result > 0

    def test_benchmark_for_list(self, benchmark):
        """Benchmark for loop over list."""
        lst = list(range(10000))

        def for_list():
            total = 0
            for item in lst:
                total += item
            return total

        result = benchmark(for_list)
        assert result > 0

    def test_benchmark_enumerate(self, benchmark):
        """Benchmark enumerate()."""
        lst = list(range(10000))

        def with_enumerate():
            total = 0
            for i, item in enumerate(lst):
                total += i + item
            return total

        result = benchmark(with_enumerate)
        assert result > 0

    def test_benchmark_zip(self, benchmark):
        """Benchmark zip()."""
        lst1 = list(range(10000))
        lst2 = list(range(10000, 20000))

        def with_zip():
            total = 0
            for a, b in zip(lst1, lst2):
                total += a + b
            return total

        result = benchmark(with_zip)
        assert result > 0

    def test_benchmark_map(self, benchmark):
        """Benchmark map()."""

        def with_map():
            return sum(map(lambda x: x * 2, range(10000)))

        result = benchmark(with_map)
        assert result > 0

    def test_benchmark_filter(self, benchmark):
        """Benchmark filter()."""

        def with_filter():
            return sum(filter(lambda x: x % 2 == 0, range(10000)))

        result = benchmark(with_filter)
        assert result > 0

    def test_benchmark_generator_expression(self, benchmark):
        """Benchmark generator expression."""

        def gen_expr():
            return sum(x * 2 for x in range(10000))

        result = benchmark(gen_expr)
        assert result > 0


class TestStringOperations:
    """Test string operation performance."""

    def test_benchmark_string_concat_plus(self, benchmark):
        """Benchmark string concatenation with +."""

        def concat_plus():
            s = ""
            for i in range(100):
                s = s + str(i)
            return len(s)

        result = benchmark(concat_plus)
        assert result > 0

    def test_benchmark_string_join(self, benchmark):
        """Benchmark string join (efficient pattern)."""

        def string_join():
            return "".join(str(i) for i in range(100))

        result = benchmark(string_join)
        assert len(result) > 0

    def test_benchmark_fstring_format(self, benchmark):
        """Benchmark f-string formatting."""

        def fstring_format():
            results = []
            for i in range(1000):
                results.append(f"value_{i}_data_{i*2}")
            return results

        result = benchmark(fstring_format)
        assert len(result) == 1000

    def test_benchmark_string_methods(self, benchmark):
        """Benchmark string methods."""
        s = "hello world " * 100

        def string_methods():
            result = s.upper()
            result = result.lower()
            result = result.replace("world", "python")
            result = result.strip()
            return len(result)

        result = benchmark(string_methods)
        assert result > 0
