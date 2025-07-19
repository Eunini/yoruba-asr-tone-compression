"""Cover each sample once without a sub-second tail at a 30-second boundary."""


def chunks(signal, maximum=480000, minimum_tail=16000):
    if maximum < minimum_tail or minimum_tail < 1 or not len(signal):
        raise ValueError("Require nonempty audio and valid chunk bounds")
    start = 0
    while len(signal) - start > maximum:
        remaining = len(signal) - start
        size = maximum if remaining - maximum >= minimum_tail else remaining - minimum_tail
        yield signal[start : start + size]
        start += size
    yield signal[start:]
