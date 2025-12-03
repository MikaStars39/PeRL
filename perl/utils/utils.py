import fire
from types import SimpleNamespace
import sys

from perl.config.config import TrainConfig

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def parse_args_to_config():
    """Parse command line arguments and create TrainConfig"""
    config = TrainConfig()

    # Parse --config.* arguments
    args = sys.argv[1:]  # Skip script name
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--config.'):
            # Parse nested config arguments like --config.common.seed 42
            config_path = arg[len('--config.'):]  # e.g., "common.seed"
            if i + 1 < len(args):
                value_str = args[i + 1]
                # Try to parse the value
                try:
                    # Try int
                    value = int(value_str)
                except ValueError:
                    try:
                        # Try float
                        value = float(value_str)
                    except ValueError:
                        # Try boolean
                        if value_str.lower() in ('true', 'false'):
                            value = value_str.lower() == 'true'
                        else:
                            # Try to parse as list (for target_modules, report_to)
                            if value_str.startswith('[') and value_str.endswith(']'):
                                import ast
                                value = ast.literal_eval(value_str)
                            else:
                                # Default to string
                                value = value_str

                # Set the nested attribute
                parts = config_path.split('.')
                obj = config
                for part in parts[:-1]:
                    if isinstance(obj, dict):
                        # Handle dict attributes (e.g., lr_scheduler_kwargs)
                        if part not in obj:
                            obj[part] = {}
                        obj = obj[part]
                    else:
                        # Handle object attributes
                        if not hasattr(obj, part):
                            raise ValueError(f"Unknown config section: {part}")
                        obj = getattr(obj, part)
                
                attr_name = parts[-1]
                if isinstance(obj, dict):
                    # Set value in dict
                    obj[attr_name] = value
                else:
                    # Set attribute on object
                    if not hasattr(obj, attr_name):
                        raise ValueError(f"Unknown config attribute: {attr_name}")
                    setattr(obj, attr_name, value)
                i += 2  # Skip the value
            else:
                raise ValueError(f"Missing value for {arg}")
        else:
            i += 1

    return config
