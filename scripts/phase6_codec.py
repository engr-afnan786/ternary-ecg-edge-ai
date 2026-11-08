"""Least-significant digit first. Stored digit = ternary code + 1."""
from itertools import product

FORMATS = {"two_bit": (4, 4), "five_trit": (3, 5)}


def pack(codes, fmt):
    base, width = FORMATS[fmt]
    data = bytearray()
    for start in range(0, len(codes), width):
        value = 0
        for place, code in enumerate(codes[start:start + width]):
            if code not in (-1, 0, 1):
                raise ValueError("Invalid ternary code")
            value += (int(code) + 1) * base ** place
        data.append(value)
    return bytes(data)


def unpack(data, count, fmt):
    base, width = FORMATS[fmt]
    if count < 0 or len(data) != (count + width - 1) // width:
        raise ValueError("Packed length mismatch")
    codes = []
    for value in data:
        for _ in range(min(width, count - len(codes))):
            value, digit = divmod(value, base)
            if digit > 2:
                raise ValueError("Reserved digit")
            codes.append(digit - 1)
        if value:
            raise ValueError("Invalid high digits")
    return codes


def self_test():
    for fmt, (base, width) in FORMATS.items():
        for length in range(width + 1):
            for group in product((-1, 0, 1), repeat=length):
                codes = list(group)
                if unpack(pack(codes, fmt), length, fmt) != codes:
                    raise ValueError("Round-trip failed")
        codes = [1, -1, 0] * 7
        if unpack(pack(codes, fmt), len(codes), fmt) != codes:
            raise ValueError("Cross-byte failure")
        if pack([-1, 0, 1], fmt) != bytes([base + 2 * base**2]):
            raise ValueError("Digit order mismatch")
    print("Packing self-test PASS")


if __name__ == "__main__":
    self_test()
