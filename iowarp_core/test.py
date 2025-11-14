#!/usr/bin/env python3
"""
IOWarp Core Test Runner

This module provides utilities for running IOWarp Core tests.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import click
from .config import IOWarpConfig


class IOWarpTestRunner:
    """Test runner for IOWarp Core."""
    
    def __init__(self, prefix: Optional[str] = None):
        self.config = IOWarpConfig(prefix=prefix)
        self.config_data = self.config.load_config()
    
    def is_installed(self) -> bool:
        """Check if IOWarp Core is installed."""
        return self.config.is_installed()
    
    def get_build_dir(self) -> Optional[Path]:
        """Get the build directory path."""
        build_dir = self.config_data.get('build_dir')
        if build_dir and Path(build_dir).exists():
            return Path(build_dir)
        return None
    
    def get_test_executable(self) -> Optional[Path]:
        """Get path to the test executable."""
        if not self.is_installed():
            return None
        
        # Try different possible locations
        test_paths = [
            self.config.prefix / "bin" / "unit_tests",
            self.config.prefix / "bin" / "iowarp_tests",
        ]
        
        build_dir = self.get_build_dir()
        if build_dir:
            test_paths.extend([
                build_dir / "bin" / "unit_tests",
                build_dir / "bin" / "iowarp_tests",
            ])
        
        for path in test_paths:
            if path.exists() and path.is_file():
                return path
        
        return None
    
    def run_unit_tests(self, test_filter: Optional[str] = None, 
                      verbose: bool = False) -> bool:
        """Run unit tests."""
        test_exe = self.get_test_executable()
        
        if not test_exe:
            click.echo("❌ Test executable not found")
            click.echo("Make sure IOWarp Core is installed with tests enabled")
            return False
        
        cmd = [str(test_exe)]
        
        if test_filter:
            cmd.extend(["--gtest_filter", test_filter])
        
        if verbose:
            cmd.append("--gtest_verbose")
        
        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode == 0
        except Exception as e:
            click.echo(f"❌ Error running tests: {e}")
            return False
    
    def run_ctest(self, test_pattern: Optional[str] = None, 
                 verbose: bool = False) -> bool:
        """Run tests using CTest."""
        build_dir = self.get_build_dir()
        
        if not build_dir:
            click.echo("❌ Build directory not found")
            return False
        
        cmd = ["ctest"]
        
        if test_pattern:
            cmd.extend(["-R", test_pattern])
        
        if verbose:
            cmd.append("-VV")
        else:
            cmd.append("-V")
        
        try:
            result = subprocess.run(cmd, cwd=build_dir, check=False)
            return result.returncode == 0
        except Exception as e:
            click.echo(f"❌ Error running CTest: {e}")
            return False
    
    def list_available_tests(self) -> List[str]:
        """List available tests."""
        build_dir = self.get_build_dir()
        
        if not build_dir:
            return []
        
        try:
            result = subprocess.run(
                ["ctest", "--show-only=json-v1"], 
                cwd=build_dir, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            import json
            test_data = json.loads(result.stdout)
            tests = []
            
            for test in test_data.get('tests', []):
                tests.append(test.get('name', ''))
            
            return tests
        except Exception:
            return []
    
    def run_component_tests(self, component: str, verbose: bool = False) -> bool:
        """Run tests for a specific component."""
        component_patterns = {
            'runtime': 'chimaera.*|hshm.*',
            'cte': 'cte.*|wrp_cte.*',
            'cae': 'cae.*|wrp_cae.*|omni.*',
            'cee': 'cee.*|wrp_cee.*',
            'transport': 'hshm.*',
        }
        
        pattern = component_patterns.get(component.lower())
        if not pattern:
            click.echo(f"❌ Unknown component: {component}")
            click.echo(f"Available components: {', '.join(component_patterns.keys())}")
            return False
        
        return self.run_ctest(test_pattern=pattern, verbose=verbose)


def run_tests(component: Optional[str] = None, test_filter: Optional[str] = None,
             verbose: bool = False, prefix: Optional[str] = None) -> bool:
    """Run IOWarp Core tests."""
    runner = IOWarpTestRunner(prefix=prefix)
    
    if not runner.is_installed():
        click.echo("❌ IOWarp Core is not installed")
        return False
    
    if component:
        return runner.run_component_tests(component, verbose=verbose)
    elif test_filter:
        return runner.run_ctest(test_pattern=test_filter, verbose=verbose)
    else:
        return runner.run_ctest(verbose=verbose)


@click.command()
@click.option('--component', '-c', default=None,
              type=click.Choice(['runtime', 'cte', 'cae', 'cee', 'transport']),
              help='Run tests for specific component')
@click.option('--filter', '-f', default=None,
              help='Test name filter pattern')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
@click.option('--list', '-l', 'list_tests', is_flag=True,
              help='List available tests')
@click.option('--prefix', '-p', default=None,
              help='Installation prefix')
def main(component: Optional[str], filter: Optional[str], verbose: bool,
         list_tests: bool, prefix: Optional[str]):
    """Run IOWarp Core tests."""
    runner = IOWarpTestRunner(prefix=prefix)
    
    if not runner.is_installed():
        click.echo("❌ IOWarp Core is not installed")
        click.echo("Run 'iowarp-install' first to install IOWarp Core")
        sys.exit(1)
    
    if list_tests:
        tests = runner.list_available_tests()
        if tests:
            click.echo("Available tests:")
            for test in tests:
                click.echo(f"  - {test}")
        else:
            click.echo("No tests found or CTest not available")
        return
    
    success = run_tests(
        component=component, 
        test_filter=filter, 
        verbose=verbose, 
        prefix=prefix
    )
    
    if success:
        click.echo("✅ Tests completed successfully")
    else:
        click.echo("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()