import sys
import os
import re

class DualLogger:
    """
    Redirects stdout to both the console and a file.
    Optionally strips ANSI escape sequences when writing to the file
    so that the file remains readable while the terminal stays colorful.
    """
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Clear or create file
        self.log_file = open(filepath, "w", encoding="utf-8")
        # Regular expression to match ANSI escape sequences (colors)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message: str):
        # Write to console (with colors)
        self.terminal.write(message)
        # Strip colors and write to log file
        clean_message = self.ansi_escape.sub('', message)
        self.log_file.write(clean_message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __del__(self):
        try:
            self.log_file.close()
        except Exception:
            pass

def setup_logger(log_path: str = "traces/reasoning_logs.txt") -> DualLogger:
    """
    Replaces sys.stdout with a DualLogger instance.
    """
    logger = DualLogger(log_path)
    sys.stdout = logger
    return logger
