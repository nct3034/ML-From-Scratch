import os
from datetime import datetime

class TrainingTracker:
    def __init__(self, debug_mode=False, log_file="logs/training_log.txt", header=""):
        self.debug_mode = debug_mode
        self.log_file = log_file
        
        # Write the custom header passed by the specific model
        if self.debug_mode:
            os.makedirs("logs", exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"--- Training Started: {datetime.now()} ---\n")
                if header:
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

    def log_step(self, *args):
        """
        Logs a generic row of data dynamically.
        Uses *args to accept any number of variables.
        """
        if not self.debug_mode:
            return
            
        # Convert all arguments to strings with fixed padding (e.g., 12 spaces) for alignment
        log_line = " | ".join(f"{str(item):<12}" for item in args) + "\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)