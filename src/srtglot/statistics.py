from dataclasses import dataclass
from functools import wraps


@dataclass
class Statistics:
    translate_batch: int = 0
    translate_batch_retried: int = 0

    def register_retry(self, name: str):
        def wrap(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if "attempt_number" in kwargs:
                    attempt_number = kwargs["attempt_number"]
                    if attempt_number > 1:
                        setattr(
                            self,
                            f"{name}_retried",
                            getattr(self, f"{name}_retried") + 1,
                        )
                    else:
                        setattr(self, name, getattr(self, name) + 1)

                return fn(*args, **kwargs)

            return wrapper

        return wrap
