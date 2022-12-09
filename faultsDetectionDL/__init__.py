import logging

package_logger_name = "faultDetectionDL"
_logger = logging.getLogger(package_logger_name)

def configure_logging(log_level: int):
    """Configure logging messages for this program."""

    # disable default logging handlers
    _logger.handlers = []
    _logger.propagate = False
    _logger.setLevel(log_level)

    console_log_handler = logging.StreamHandler()
    if log_level <= logging.DEBUG:
        console_formatter = logging.Formatter(
            fmt="[%(asctime)s] - %(levelname)s (from %(name)s in %(filename)s:%(lineno)d): %(message)s"
        )
    else:
        console_formatter = logging.Formatter(fmt="[%(asctime)s] - %(levelname)s: %(message)s")

    console_log_handler.setFormatter(console_formatter)
    console_log_handler.setLevel(log_level)
    _logger.addHandler(console_log_handler)
    return

configure_logging(logging.INFO)