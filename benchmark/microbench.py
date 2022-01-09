import datetime
import itertools
import operator
import statistics
import timeit
from collections import defaultdict

from decimal import Decimal
from fractions import Fraction as PyFraction
from quicktions import Fraction as QFraction

PyFraction.__name__ = "PyFraction"

class_names, classes = list(zip(*[
    ('float', float),
    ('Decimal', Decimal),
    ('QFraction', QFraction),
    ('PyFraction', PyFraction),
]))

benchmark_values = [
    "123",
    "76138945735803",
    "123.456789",
    "1234e10",
    "1234e-10",
    "983675298736458927345e10",
    "983675298736458927345e-10",
    #"983675298736458927345e20",
    #"983675298736458927345e-20",
    "983675298736458927345e50",
    "983675298736458927345e-50",
]


benchmark_expressions = [
    "a + b + c",
    "a - b - c",
    "a * b * c",
    "a / b / c",
    "a * 10 + b * 99 + c",
    "a * 10 * b + c",
    "a / 10 / b - c",
    "a * 32543575324 * b / c",
    "(a * 32543575324) / (b * 65767564) * c",
    "a ** 3 + b ** 2 + c ** 1",
]


def all_types(values, for_cls):
    # "123.456" => "'123.456'" -- "123.456" -- "123456, 1000"
    if for_cls in (PyFraction, QFraction):
        def int_args(a, b):
            return f"{a}, {b}"
        def types(v):
            f = QFraction(v).as_integer_ratio()
            return (
                repr(v),
                v,
                int_args(f[0], f[1]),
                int_args(f[0]*10, f[1]*100),
                int_args(f[0]*99, f[1]*f[0] - 1),
                int_args(f[0]*64*64, f[1]*1024*1024),
            )
    else:
        def types(v):
            return (
                repr(v),
                v,
            )

    return [
        tp_v
        for v in values
        for tp_v in types(v)
    ]


def run_bm(code, setup='pass', number=200, repeat=1_000, gns=None):
    times = timeit.repeat(code, setup, number=number, repeat=repeat, globals=gns)
    p10 = len(times) // 10
    if p10 >= 1:
        times.sort()
        times = times[p10:-p10]
    return statistics.mean(times) * 1_000_000  # s -> us


def bm_instantiation(values=benchmark_values, classes=classes):
    gns = {}
    for cls in classes:
        cls_name = cls.__name__
        gns[cls_name] = cls
        results = {}
        for args in all_types(values, cls):
            code = f"{cls_name}({args})"
            results[code] = run_bm(code, gns=gns)
        yield (cls, results)


def bm_calculation(expressions=benchmark_expressions, values=benchmark_values, classes=classes):
    gns = {}
    for cls in classes:
        results = defaultdict(list)
        cls_name = cls.__name__
        for a, b, c in itertools.product(values, repeat=3):
            gns.update(a=cls(a), b=cls(b), c=cls(c))

            for expr in expressions:
                results[f"{cls_name}: {expr}"].append(run_bm(expr, number=30, repeat=100, gns=gns))

        # average over all value combinations
        results = {expr: statistics.mean(t) for expr, t in results.items()}
        yield (cls, results)


def run_benchmarks():
    for what, bm in [
        ('create', bm_instantiation()),
        ('compute', bm_calculation()),
    ]:
        for cls, result in bm:
            yield ((what, cls), result)


def main():
    results_by_type = defaultdict(dict)
    now = datetime.datetime.now
    start_time = now()
    print(start_time)
    for key, res in run_benchmarks():
        for code, t in sorted(res.items()):
            print(f"{code:50s}: {t:8.2f} us")
        results_by_type[key].update(res)
        print("++", datetime.timedelta(seconds=(now() - start_time).seconds))

    avg_by_type = defaultdict(dict)
    for (what, cls), results in results_by_type.items():
        avg_by_type[what][cls] = statistics.mean(results.values())

    for what, timings in sorted(avg_by_type.items(), reverse=True):
        print()
        print(f"Average times for all '{what}' benchmarks:")
        min_avg = min(timings.values())
        for cls, avg in sorted(timings.items(), key=operator.itemgetter(1)):
            print(f"{cls.__name__:20s}: {avg:8.2f} us ({avg / min_avg:.1f}x)")


if __name__ == '__main__':
    main()
