import warnings


def custom_formatwarning(
    msg, category, filename="", lineno=-1, module="", *args, **kwargs
):
    # ignore everything except the message
    # lineno = kwargs['lineno']
    # print(lineno)
    # print(category)
    return f"  File '{filename}', line {lineno}\n    {msg}\n"


warnings.formatwarning = custom_formatwarning
