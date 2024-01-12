import warnings


def custom_formatwarning(
    msg, category, filename="", lineno=-1, module="", *args, **kwargs
):
    """ A way to print out only certain parts of the standard warning message """
    return f"  File '{filename}', line {lineno}\n    {msg}\n"


warnings.formatwarning = custom_formatwarning
