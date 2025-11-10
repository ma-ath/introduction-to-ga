import sys
import logging
import requests
import json
import html
from pathlib import Path
from datetime import datetime
from typing import Final, Optional
from string import Template


class TelegramLogsHandler(logging.Handler):
    def __init__(self,
                 telegram_api_token: str,
                 telegram_user_id: str,
                 telegram_api_url_template: Template = Template("https://api.telegram.org/bot${token}/sendMessage")
                 ) -> None:
        self._telegram_user_id = telegram_user_id
        self._telegram_api_url = telegram_api_url_template.substitute(token=telegram_api_token)
        print(f"Telegram API URL: {self._telegram_api_url}")
        super(TelegramLogsHandler, self).__init__()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            payload = {
                'chat_id': self._telegram_user_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            response = requests.post(self._telegram_api_url, data=payload)

            if not response.ok:
                logging.getLogger(__name__).debug(f"Telegram API error: {response.status_code}")

            response_json = response.json()
            assert response_json['ok'] is True

        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to send log to Telegram API: {e!r}")


class TelegramLogsFormatter(logging.Formatter):
    """
    Formats log records for Telegram API messages.
    """
    EMOJIS: Final[dict[str, str]] = {
        "DEBUG": "🐞",
        "INFO": "👀",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🤬",
        "NOTSET": "❓"
    }

    def format(self, record: logging.LogRecord) -> str:
        message = html.escape(record.getMessage())
        emoji = self.EMOJIS.get(record.levelname, "🔍")
        t = datetime.now().strftime("%Y/%m/%d-%H:%M")

        return (
            f"<i>{t}</i> <b>{record.levelname}</b><tg-emoji emoji-id=\"5368324170671202286\">{emoji}</tg-emoji>"
            f"<pre>{message}</pre>"
        )


def setup_logger(
        name: str = __name__,
        console_level: Optional[str] = "INFO",
        file_level: Optional[str] = "DEBUG",
        telegram_level: Optional[str] = None,
        telegram_secrets: Optional[Path] = None,
    ) -> logging.Logger:
    """
    Sets up a logger that can log to console, file and a Telegram API.
    :param name: Name of the logger.
    :param console_level: Logging level for console output. If None, console logging is disabled.
    :param file_level: Logging level for file output. If None, file logging is disabled.
    :param telegram_level: Logging level for Telegram API output. If None, Telegram logging is disabled.

    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s][%(name)s][%(levelname)s]  %(message)s"
    )

    assert console_level is None or console_level.upper() in logging._nameToLevel
    assert file_level is None or file_level.upper() in logging._nameToLevel
    assert telegram_level is None or telegram_level.upper() in logging._nameToLevel

    # Console handler
    if console_level is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(console_level.upper())
        logger.addHandler(console)

    # File handler
    if file_level is not None:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.log")
        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)

    # Telegram handler
    if telegram_level is not None:
        assert telegram_secrets is not None, "telegram_secrets path must be provided for Telegram logging."
        assert telegram_secrets.exists(), f"telegram_secrets file not found at {telegram_secrets!r}"
        with open(telegram_secrets) as f:
            secrets = json.load(f)
            telegram_api_token = secrets['api_token']
            telegram_user_id = secrets['user_id']
        telegram_handler = TelegramLogsHandler(telegram_api_token, telegram_user_id)
        telegram_handler.setFormatter(TelegramLogsFormatter())
        telegram_handler.setLevel(telegram_level.upper())
        logger.addHandler(telegram_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger(
        name=__name__,
        console_level="INFO",
        file_level="DEBUG",
        telegram_level="ERROR",
        telegram_secrets=Path.home() / "Secrets" / "telegram_api.json"
    )

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
