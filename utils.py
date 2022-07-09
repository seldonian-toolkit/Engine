def error(output_string):
    raise Exception(output_string)


def clamp(val_in, min_val, max_val):
    if min_val > max_val:
        error(f"min_val {min_val} > max_val {max_val}")
    return min(max(val_in, min_val), max_val)
