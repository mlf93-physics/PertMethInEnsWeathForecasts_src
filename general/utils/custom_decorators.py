def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        # Return the function decorated
        return dec(func)

    return decorator
