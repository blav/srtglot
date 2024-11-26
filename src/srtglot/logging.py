from collections.abc import Callable
from logging import FileHandler, NullHandler, getLogger

from .config import Config
from .prompt import UserPrompt


def setup_llm_logging(config: Config) -> Callable[[UserPrompt, str | None], None]:
    llm_handler = (
        FileHandler(filename=config.llm_log_dir / "llm.log")
        if config.llm_log_dir
        else NullHandler()
    )

    llm_logger = getLogger("llm")
    llm_logger.addHandler(llm_handler)
    llm_logger.setLevel("DEBUG")

    def logger(prompt: UserPrompt, completion: str | None) -> None:
        llm_logger.debug("========================================")
        llm_logger.debug(prompt.user_message.get("content"))
        llm_logger.debug("----------------------------------------")
        llm_logger.debug(completion)
        llm_logger.debug("========================================")

    return logger
