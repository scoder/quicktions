#-*- coding: UTF-8 -*-

# The MIT License
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Adapted for fractions by Stefan Behnel.  Original version from
# https://hg.python.org/benchmarks/file/9a1136898539/performance/bm_telco.py


""" Telco Benchmark for measuring the performance of Fraction calculations

http://www2.hursley.ibm.com/decimal/telco.html
http://www2.hursley.ibm.com/decimal/telcoSpec.html

A call type indicator, c, is set from the bottom (least significant) bit of the duration (hence c is 0 or 1).
A r, r, is determined from the call type. Those calls with c=0 have a low r: 0.0013; the remainder (‘distance calls’) have a ‘premium’ r: 0.00894. (The rates are, very roughly, in Euros or dollarates per second.)
A price, p, for the call is then calculated (p=r*n).
A basic tax, b, is calculated: b=p*0.0675 (6.75%), and the total basic tax variable is then incremented (sumB=sumB+b).
For distance calls: a distance tax, d, is calculated: d=p*0.0341 (3.41%), and then the total distance tax variable is incremented (sumD=sumD+d).
The total price, t, is calculated (t=p+b, and, if a distance call, t=t+d).
The total prices variable is incremented (sumT=sumT+t).
The total price, t, is converted to a string, s.
"""

import os
from struct import unpack
from time import time


def rel_path(*path):
    return os.path.join(os.path.dirname(__file__), *path)

try:
    from quicktions import Fraction
except ImportError:
    import sys
    sys.path.insert(0, rel_path('..', 'src'))

from quicktions import Fraction


filename = rel_path("telco-bench.b")


def run(cls):
    rates = list(map(cls, ('0.0013', '0.00894')))
    basictax = cls("0.0675")
    disttax = cls("0.0341")

    values = []
    with open(filename, "rb") as infil:
        for _ in range(20000):
            datum = infil.read(8)
            if datum == '': break
            n, =  unpack('>Q', datum)
            values.append(n)

    start = time()

    sumT = cls()   # sum of total prices
    sumB = cls()   # sum of basic tax
    sumD = cls()   # sum of 'distance' tax

    for n in values:
        calltype = n & 1
        r = rates[calltype]

        p = r * n
        b = p * basictax
        sumB += b
        t = p + b

        if calltype:
            d = p * disttax
            sumD += d
            t += d

        sumT += t

    end = time()
    return end - start


def main(n, cls=Fraction):
    run(cls)  # warmup
    times = []
    for _ in range(n):
        times.append(run(cls))
    return times


if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser(
        usage="%prog [options]",
        description="Test the performance of the Telco fractions benchmark")
    parser.add_option("-n", "--num_runs", action="store", type="int", default=12,
                      dest="num_runs", help="Number of times to repeat the benchmark.")
    parser.add_option("--use-decimal", action="store_true", default=False,
                      dest="use_decimal", help="Run benchmark with Decimal instead of Fraction.")
    parser.add_option("--use-stdlib", action="store_true", default=False,
                      dest="use_stdlib", help="Run benchmark with fractions.Fraction from stdlib.")
    options, args = parser.parse_args()

    num_class = Fraction
    if options.use_decimal:
        from decimal import Decimal as num_class
    elif options.use_stdlib:
        from fractions import Fraction as num_class

    results = main(options.num_runs, num_class)
    for result in results:
        print(result)
