
import sys
import unicodedata
from collections import defaultdict


def list_digits():
    category_of = unicodedata.category
    return [
        ch for ch in range(128, 1114111+1)
        if category_of(chr(ch)) == 'Nd'
    ]


def map_to_ascii_digit(digits):
    adigit_to_udigit = defaultdict(list)
    for ch in digits:
        adigit = int(chr(ch))
        if ch & 15 == adigit:
            adigit_to_udigit['{digit} & 15'].append(ch)
        elif (ch - 6) & 15 == adigit:
            adigit_to_udigit['({digit} - 6) & 15'].append(ch)
        elif (ch - 0x116da) & 15 == adigit:
            adigit_to_udigit['({digit} - 0x116da) & 15'].append(ch)
        elif (ch - 0x1e5f1) & 15 == adigit:
            adigit_to_udigit['({digit} - 0x1e5f1) & 15'].append(ch)
        elif ch >= 0x1d7ce and (ch - 0x1d7ce) % 10 == adigit:
            adigit_to_udigit['({digit} - 0x1d7ce) % 10'].append(ch)
        else:
            adigit_to_udigit[str(adigit)].append(ch)
    return adigit_to_udigit


def gen_switch_cases(digits_by_adigit):
    pyver = sys.version_info
    print(f"/* Switch cases generated from Python {pyver[0]}.{pyver[1]}.{pyver[2]} {pyver[3]} {pyver[4]} */")
    print("/* Deliberately excluding ASCII digit characters. */")
    print()
    print(f"/* Lowest digit: {min(ch for characters in digits_by_adigit.values() for ch in characters)} */")
    print("switch (digit) {")

    for adigit_format, udigits in digits_by_adigit.items():
        for ch in udigits:
            print(f'    case 0x{ch:x}:')

        print(f"        return {adigit_format.format(digit='digit')};")

    print("    default:")
    print("        return -1;")
    print("}")

if __name__ == '__main__':
    gen_switch_cases(map_to_ascii_digit(list_digits()))
