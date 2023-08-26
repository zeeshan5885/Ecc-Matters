# The below code was heavily influenced from:
# https://code.activestate.com/recipes/577659-decorators-for-adding-aliases-to-methods-in-a-clas/
# , and was modified for ligo-gracedb.

# Please see the referenced page for authorship and copyright information.

import warnings
import functools


class alias(object):
    """
    Alias class that can be used as a decorator for making methods callable
    through other names (or "aliases").
    """

    def __init__(self, *aliases):
        self.aliases = set(aliases)

    def __call__(self, f):
        """
        Method call wrapper.
        """
        f._aliases = self.aliases
        return f


def aliased(aliased_class):
    """
    Decorator function that must be used in combination with @alias
    decorator. Modified to produce a deprecation warning.
    """
    def warning_wrapper(func, alias):

        warning_string = ("Method {old_name} has been replaced by {new_name}"
                          ", and will be deprecated in a future release")

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            warnings.warn(warning_string.format(old_name=alias,
                                                new_name=func.__name__),
                          DeprecationWarning,
                          stacklevel=2)
            return func(*args, **kwds)
        return wrapper

    original_methods = aliased_class.__dict__.copy()
    for name, method in original_methods.items():
        if hasattr(method, '_aliases'):
            for alias in method._aliases - set(original_methods):
                wrapped_method = warning_wrapper(method, alias)
                setattr(aliased_class, alias, wrapped_method)
    return aliased_class
