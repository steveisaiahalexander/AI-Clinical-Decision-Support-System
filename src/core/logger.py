"""
Logger Configuration
====================

Centralized logging system for the AI Clinical Decision Support System.

Every module should import `logger` from this file instead of
using print().
"""

import logging
import sys

from colorlog import ColoredFormatter

from src.core.config import Config


class Logger:

    @staticmethod
    def get_logger(name: str = "AI-CDSS") -> logging.Logger:

        logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)

        file_handler = logging.FileHandler(
            Config.LOGS_DIR / "application.log",
            encoding="utf-8"
        )

        console_formatter = ColoredFormatter(

            "%(log_color)s"

            "[%(asctime)s] "

            "[%(levelname)s] "

            "%(message)s",

            datefmt="%H:%M:%S",

            log_colors={

                "DEBUG": "cyan",

                "INFO": "green",

                "WARNING": "yellow",

                "ERROR": "red",

                "CRITICAL": "bold_red",

            },

        )

        file_formatter = logging.Formatter(

            "[%(asctime)s] "

            "[%(levelname)s] "

            "%(message)s"

        )

        console_handler.setFormatter(console_formatter)

        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)

        logger.addHandler(file_handler)

        return logger


logger = Logger.get_logger()