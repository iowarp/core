#!/usr/bin/env python3
"""
IOWarp Core Configuration Management

This module provides utilities for managing IOWarp Core configuration.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import click


class IOWarpConfig:
    """Configuration manager for IOWarp Core."""
    
    def __init__(self, prefix: Optional[str] = None):
        self.prefix = Path(prefix) if prefix else Path.home() / ".local"
        self.config_file = self.prefix / "etc" / "iowarp" / "config.json"
        self.env_script = self.prefix / "bin" / "iowarp-env.sh"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            click.echo(f"Error loading config: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except IOError as e:
            click.echo(f"Error saving config: {e}")
            return False
    
    def get_value(self, key: str) -> Any:
        """Get a configuration value."""
        config = self.load_config()
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None
    
    def set_value(self, key: str, value: Any) -> bool:
        """Set a configuration value."""
        config = self.load_config()
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        return self.save_config(config)
    
    def is_installed(self) -> bool:
        """Check if IOWarp Core is installed."""
        config = self.load_config()
        return bool(config.get('version'))
    
    def get_install_info(self) -> Dict[str, Any]:
        """Get installation information."""
        config = self.load_config()
        return {
            'installed': self.is_installed(),
            'version': config.get('version', 'Unknown'),
            'prefix': config.get('prefix', str(self.prefix)),
            'build_type': config.get('build_type', 'Unknown'),
            'components': config.get('installed_components', {}),
        }
    
    def get_environment_vars(self) -> Dict[str, str]:
        """Get environment variables for IOWarp Core."""
        return {
            'IOWARP_PREFIX': str(self.prefix),
            'IOWARP_ROOT': str(self.prefix),
            'PATH': f"{self.prefix}/bin:{os.environ.get('PATH', '')}",
            'LD_LIBRARY_PATH': f"{self.prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
            'PKG_CONFIG_PATH': f"{self.prefix}/lib/pkgconfig:{os.environ.get('PKG_CONFIG_PATH', '')}",
            'CMAKE_MODULE_PATH': f"{self.prefix}/lib/cmake:{os.environ.get('CMAKE_MODULE_PATH', '')}",
        }
    
    def show_status(self) -> None:
        """Display installation status."""
        info = self.get_install_info()
        
        click.echo("IOWarp Core Status:")
        click.echo(f"  Installed: {'✅ Yes' if info['installed'] else '❌ No'}")
        
        if info['installed']:
            click.echo(f"  Version: {info['version']}")
            click.echo(f"  Prefix: {info['prefix']}")
            click.echo(f"  Build Type: {info['build_type']}")
            click.echo("  Components:")
            for component, enabled in info['components'].items():
                status = "✅" if enabled else "❌"
                click.echo(f"    {status} {component}")
            
            click.echo(f"\nEnvironment script: {self.env_script}")
            if self.env_script.exists():
                click.echo("  Run: source ~/.local/bin/iowarp-env.sh")
            else:
                click.echo("  ❌ Environment script not found")


def get_config(key: Optional[str] = None, prefix: Optional[str] = None) -> Any:
    """Get configuration value(s)."""
    config_mgr = IOWarpConfig(prefix=prefix)
    
    if key is None:
        return config_mgr.load_config()
    else:
        return config_mgr.get_value(key)


def set_config(key: str, value: Any, prefix: Optional[str] = None) -> bool:
    """Set configuration value."""
    config_mgr = IOWarpConfig(prefix=prefix)
    return config_mgr.set_value(key, value)


@click.group()
@click.option('--prefix', '-p', default=None,
              help='Configuration prefix (default: ~/.local)')
@click.pass_context
def main(ctx: click.Context, prefix: Optional[str]):
    """IOWarp Core configuration management."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = IOWarpConfig(prefix=prefix)


@main.command()
@click.pass_context
def status(ctx: click.Context):
    """Show installation status."""
    config_mgr = ctx.obj['config']
    config_mgr.show_status()


@main.command()
@click.argument('key')
@click.pass_context
def get(ctx: click.Context, key: str):
    """Get a configuration value."""
    config_mgr = ctx.obj['config']
    value = config_mgr.get_value(key)
    
    if value is None:
        click.echo(f"Key '{key}' not found")
        sys.exit(1)
    else:
        if isinstance(value, (dict, list)):
            click.echo(json.dumps(value, indent=2))
        else:
            click.echo(str(value))


@main.command()
@click.argument('key')
@click.argument('value')
@click.pass_context
def set(ctx: click.Context, key: str, value: str):
    """Set a configuration value."""
    config_mgr = ctx.obj['config']
    
    # Try to parse value as JSON, fall back to string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value
    
    if config_mgr.set_value(key, parsed_value):
        click.echo(f"Set {key} = {parsed_value}")
    else:
        click.echo(f"Failed to set {key}")
        sys.exit(1)


@main.command()
@click.pass_context
def env(ctx: click.Context):
    """Show environment variables."""
    config_mgr = ctx.obj['config']
    env_vars = config_mgr.get_environment_vars()
    
    for key, value in env_vars.items():
        click.echo(f"export {key}=\"{value}\"")


@main.command()
@click.pass_context
def paths(ctx: click.Context):
    """Show important paths."""
    config_mgr = ctx.obj['config']
    
    click.echo("IOWarp Core Paths:")
    click.echo(f"  Config file: {config_mgr.config_file}")
    click.echo(f"  Env script: {config_mgr.env_script}")
    click.echo(f"  Prefix: {config_mgr.prefix}")
    click.echo(f"  Binaries: {config_mgr.prefix}/bin")
    click.echo(f"  Libraries: {config_mgr.prefix}/lib")
    click.echo(f"  Headers: {config_mgr.prefix}/include")


if __name__ == "__main__":
    main()