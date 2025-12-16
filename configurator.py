"""
Configuration system that supports:
1. Default values defined in main.py
2. YAML config files from configs/ directory
3. Command line argument overrides

Usage:
    python main.py --config my_config.yaml --batch_size 32 --learning_rate 1e-3
"""

import os
import sys
import yaml
import argparse
import ast


def parse_value(value_str):
    """Parse a string value to its appropriate Python type."""
    try:
        # Try to parse as Python literal (handles bool, None, numbers, etc.)
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If that fails, return as string
        return value_str


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config if config else {}


def convert_to_type(value, target_type):
    """
    Convert a value to the target type.
    
    Args:
        value: Value to convert
        target_type: Target type (int, float, bool, str, or type instance)
    
    Returns:
        Converted value, or original value if conversion fails
    """
    if isinstance(value, target_type):
        return value
    
    # Handle bool conversion
    if target_type == bool:
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    # Handle int conversion
    if target_type == int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                parsed = parse_value(value)
                return int(parsed) if isinstance(parsed, (int, float)) else value
        return value
    
    # Handle float conversion
    if target_type == float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                parsed = parse_value(value)
                return float(parsed) if isinstance(parsed, (int, float)) else value
        return value
    
    # For other types, return as-is
    return value


def apply_config_overrides(config_dict, globals_dict, original_defaults=None):
    """
    Apply configuration dictionary to global variables with type conversion.
    
    Args:
        config_dict: Dictionary of config values to apply
        globals_dict: The globals() dictionary to update
        original_defaults: Dictionary of original default values for type checking
    """
    for key, value in config_dict.items():
        if key not in globals_dict:
            print(f"Warning: Unknown config key '{key}' will be ignored")
            continue
        
        # Convert value to match the expected type from defaults
        if original_defaults and key in original_defaults:
            expected_type = type(original_defaults[key])
            value = convert_to_type(value, expected_type)
        
        globals_dict[key] = value


def _get_provided_args():
    """Extract set of command line argument keys that were explicitly provided."""
    provided_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            key = arg[2:].split('=')[0]
            provided_args.add(key)
    return provided_args


def _add_argument_for_key(parser, key, default_value):
    """Add an argument to the parser for a given config key."""
    if default_value is None:
        parser.add_argument(f'--{key}', type=str, default=None,
                          help=f'{key} (default: None)')
    elif isinstance(default_value, bool):
        parser.add_argument(f'--{key}', type=lambda x: x.lower() in ('true', '1', 'yes', 'on'),
                          default=default_value, help=f'{key} (default: {default_value})')
    elif isinstance(default_value, int):
        parser.add_argument(f'--{key}', type=int, default=default_value,
                          help=f'{key} (default: {default_value})')
    elif isinstance(default_value, float):
        parser.add_argument(f'--{key}', type=float, default=default_value,
                          help=f'{key} (default: {default_value})')
    else:  # str
        parser.add_argument(f'--{key}', type=str, default=default_value,
                          help=f'{key} (default: {default_value})')


def _resolve_config_path(config_path):
    """Resolve config file path, checking configs/ directory if not absolute."""
    if os.path.isabs(config_path):
        return config_path
    
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    return os.path.join(configs_dir, config_path)


def _load_yaml_config_safe(config_path):
    """Load YAML config file, handling path resolution and errors."""
    resolved_path = _resolve_config_path(config_path)
    
    if os.path.exists(resolved_path):
        config = load_yaml_config(resolved_path)
        print(f"Loaded config from: {resolved_path}")
        return config
    else:
        print(f"Warning: Config file not found: {resolved_path}")
        return {}


def _apply_command_line_args(args, config_keys, original_defaults, globals_dict, provided_args):
    """Apply command line arguments to globals_dict with type conversion."""
    for key in config_keys:
        if not hasattr(args, key):
            continue
        
        # Only apply if argument was explicitly provided
        if key not in provided_args:
            continue
        
        arg_value = getattr(args, key)
        expected_type = type(original_defaults[key])
        
        # Convert string arguments to appropriate types
        if isinstance(arg_value, str) and expected_type != str:
            arg_value = convert_to_type(arg_value, expected_type)
        
        globals_dict[key] = arg_value


def _create_final_config_dict(config_keys, globals_dict, original_defaults):
    """
    Create final config dictionary from globals_dict, ensuring correct types.
    
    This ensures wandb logs the actual values after all overrides.
    """
    config = {}
    for key in config_keys:
        value = globals_dict[key]
        expected_type = type(original_defaults[key])
        
        # Ensure integer values are actually integers (handle edge cases)
        if expected_type == int and not isinstance(value, int):
            try:
                value = int(value)
                globals_dict[key] = value  # Update globals_dict too
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails
        
        config[key] = value
    
    return config


def setup_config(globals_dict):
    """
    Setup configuration system.
    
    This function:
    1. Collects all configurable variables (int, float, bool, str, None)
    2. Parses command line arguments
    3. Loads YAML config if specified
    4. Applies overrides in order: defaults -> YAML -> command line
    
    Args:
        globals_dict: The globals() dictionary from the calling module
    
    Returns:
        config: Dictionary of all configuration values (for logging)
    """
    # Collect configurable keys (int, float, bool, str, None)
    # Exclude internal variables
    exclude_keys = {
        'config', 'setup_config', 'parse_value', 'load_yaml_config',
        'apply_config_overrides', 'os', 'yaml', 'argparse', 'ast', 'sys'
    }
    config_keys = [
        k for k, v in globals_dict.items()
        if (not k.startswith('_') and 
            isinstance(v, (int, float, bool, str, type(None))) and
            k not in exclude_keys)
    ]
    
    # Store original defaults for type checking
    original_defaults = {k: globals_dict[k] for k in config_keys}
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Bandformer Training Script')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (relative to configs/ directory or absolute path)')
    
    # Add arguments for all configurable variables
    for key in config_keys:
        _add_argument_for_key(parser, key, globals_dict[key])
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load YAML config if specified
    yaml_config = {}
    if args.config:
        yaml_config = _load_yaml_config_safe(args.config)
    
    # Apply YAML config first (overrides defaults)
    apply_config_overrides(yaml_config, globals_dict, original_defaults)
    
    # Track which args were explicitly provided
    provided_args = _get_provided_args()
    
    # Apply command line arguments (overrides YAML and defaults)
    _apply_command_line_args(args, config_keys, original_defaults, globals_dict, provided_args)
    
    # Create final config dict for logging (ensures correct types)
    config = _create_final_config_dict(config_keys, globals_dict, original_defaults)
    
    return config
