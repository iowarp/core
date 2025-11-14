#!/usr/bin/env python3
"""
IOWarp Main Command Line Interface

This is the main entry point for the IOWarp installer package.
"""

import os
import sys
import subprocess
import shutil
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
import psutil
import requests


class IOWarpInstaller:
    """Main installer class for IOWarp Core."""
    
    def __init__(self, prefix: Optional[str] = None, build_type: str = "Release"):
        self.prefix = Path(prefix) if prefix else Path.home() / ".local"
        self.build_type = build_type
        self.repo_url = "https://github.com/iowarp/iowarp-core.git"
        self.config_file = self.prefix / "etc" / "iowarp" / "config.json"
        
    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check if required system dependencies are available."""
        deps = {
            "cmake": self._check_command("cmake", ["--version"]),
            "gcc": self._check_command("gcc", ["--version"]),
            "g++": self._check_command("g++", ["--version"]),
            "make": self._check_command("make", ["--version"]),
            "git": self._check_command("git", ["--version"]),
            "pkg-config": self._check_command("pkg-config", ["--version"]),
        }
        
        # Check for optional dependencies
        optional_deps = {
            "mpicc": self._check_command("mpicc", ["--version"]),
            "nvcc": self._check_command("nvcc", ["--version"]),
            "hipcc": self._check_command("hipcc", ["--version"]),
        }
        
        return {"required": deps, "optional": optional_deps}
    
    def _check_command(self, cmd: str, args: List[str]) -> bool:
        """Check if a command exists and is executable."""
        try:
            result = subprocess.run(
                [cmd] + args, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def check_disk_space(self, required_gb: float = 5.0) -> bool:
        """Check if there's enough disk space for installation."""
        try:
            usage = psutil.disk_usage(str(self.prefix.parent))
            available_gb = usage.free / (1024**3)
            return available_gb >= required_gb
        except Exception:
            return True  # Assume OK if we can't check
    
    def clone_or_update_repo(self, source_dir: Path) -> bool:
        """Clone the repository or update if it already exists."""
        if source_dir.exists():
            click.echo(f"Repository already exists at {source_dir}")
            if click.confirm("Update existing repository?"):
                try:
                    subprocess.run(
                        ["git", "pull"], 
                        cwd=source_dir, 
                        check=True, 
                        capture_output=True
                    )
                    # Update submodules
                    subprocess.run(
                        ["git", "submodule", "update", "--init", "--recursive"], 
                        cwd=source_dir, 
                        check=True, 
                        capture_output=True
                    )
                    click.echo("Repository updated successfully")
                    return True
                except subprocess.CalledProcessError as e:
                    click.echo(f"Failed to update repository: {e}")
                    return False
            return True
        else:
            try:
                subprocess.run(
                    ["git", "clone", "--recursive", self.repo_url, str(source_dir)], 
                    check=True, 
                    capture_output=True
                )
                click.echo("Repository cloned successfully")
                return True
            except subprocess.CalledProcessError as e:
                click.echo(f"Failed to clone repository: {e}")
                return False
    
    def configure_cmake(self, source_dir: Path, build_dir: Path) -> bool:
        """Configure the build using CMake."""
        build_dir.mkdir(parents=True, exist_ok=True)
        
        cmake_args = [
            "cmake",
            str(source_dir),
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            f"-DCMAKE_INSTALL_PREFIX={self.prefix}",
            "-DWRP_CORE_ENABLE_RUNTIME=ON",
            "-DWRP_CORE_ENABLE_CTE=ON", 
            "-DWRP_CORE_ENABLE_CAE=ON",
            "-DWRP_CORE_ENABLE_CEE=ON",
            "-DWRP_CORE_ENABLE_TESTS=OFF",  # Disable tests for installation
            "-DWRP_CORE_ENABLE_BENCHMARKS=OFF",
            "-DWRP_CORE_ENABLE_PYTHON=ON",  # Enable Python bindings
            "-DBUILD_SHARED_LIBS=ON",
        ]
        
        try:
            subprocess.run(cmake_args, cwd=build_dir, check=True)
            click.echo("CMake configuration completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            click.echo(f"CMake configuration failed: {e}")
            return False
    
    def build_project(self, build_dir: Path) -> bool:
        """Build the project using make."""
        # Determine number of parallel jobs
        nproc = min(psutil.cpu_count(), 12)  # Limit to 12 to avoid overwhelming the system
        
        try:
            subprocess.run(
                ["cmake", "--build", ".", f"-j{nproc}"], 
                cwd=build_dir, 
                check=True
            )
            click.echo("Build completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            click.echo(f"Build failed: {e}")
            return False
    
    def install_project(self, build_dir: Path) -> bool:
        """Install the project."""
        try:
            subprocess.run(
                ["cmake", "--install", "."], 
                cwd=build_dir, 
                check=True
            )
            click.echo("Installation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            click.echo(f"Installation failed: {e}")
            return False
    
    def save_config(self, source_dir: Path, build_dir: Path) -> None:
        """Save installation configuration."""
        config = {
            "version": "1.0.0",
            "prefix": str(self.prefix),
            "build_type": self.build_type,
            "source_dir": str(source_dir),
            "build_dir": str(build_dir),
            "repo_url": self.repo_url,
            "installed_components": {
                "runtime": True,
                "cte": True,
                "cae": True,
                "cee": True,
            }
        }
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_environment(self) -> None:
        """Set up environment variables and paths."""
        # Create shell script for environment setup
        env_script = self.prefix / "bin" / "iowarp-env.sh"
        env_content = f"""#!/bin/bash
# IOWarp Core Environment Setup Script

export IOWARP_PREFIX="{self.prefix}"
export IOWARP_ROOT="{self.prefix}"

# Add to PATH
export PATH="{self.prefix}/bin:$PATH"

# Add to library path
export LD_LIBRARY_PATH="{self.prefix}/lib:$LD_LIBRARY_PATH"

# Add to pkg-config path
export PKG_CONFIG_PATH="{self.prefix}/lib/pkgconfig:$PKG_CONFIG_PATH"

# Add to CMake module path
export CMAKE_MODULE_PATH="{self.prefix}/lib/cmake:$CMAKE_MODULE_PATH"

echo "IOWarp Core environment configured successfully"
echo "Installation prefix: {self.prefix}"
"""
        
        env_script.parent.mkdir(parents=True, exist_ok=True)
        with open(env_script, 'w') as f:
            f.write(env_content)
        env_script.chmod(0o755)
        
        click.echo(f"Environment setup script created at: {env_script}")
        click.echo(f"Run 'source {env_script}' to set up your environment")
    
    def install(self, source_dir: Optional[Path] = None, keep_build: bool = False) -> bool:
        """Main installation method."""
        # Check dependencies
        deps = self.check_system_dependencies()
        missing_required = [k for k, v in deps["required"].items() if not v]
        
        if missing_required:
            click.echo("❌ Missing required dependencies:")
            for dep in missing_required:
                click.echo(f"  - {dep}")
            click.echo("\nPlease install the missing dependencies and try again.")
            click.echo("\nFor Ubuntu/Debian:")
            click.echo("sudo apt-get install build-essential cmake pkg-config \\")
            click.echo("  libboost-context-dev libboost-fiber-dev libboost-system-dev \\")
            click.echo("  libelf-dev libzmq3-dev")
            return False
        
        click.echo("✅ All required dependencies found")
        
        # Check available optional dependencies
        available_optional = [k for k, v in deps["optional"].items() if v]
        if available_optional:
            click.echo(f"📦 Optional dependencies found: {', '.join(available_optional)}")
        
        # Check disk space
        if not self.check_disk_space():
            click.echo("❌ Insufficient disk space (need at least 5GB)")
            return False
        
        # Set up directories
        if source_dir is None:
            source_dir = Path.home() / ".cache" / "iowarp-core"
            
        build_dir = source_dir / "build"
        
        try:
            # Clone or update repository
            if not self.clone_or_update_repo(source_dir):
                return False
            
            # Configure with CMake
            if not self.configure_cmake(source_dir, build_dir):
                return False
            
            # Build the project
            if not self.build_project(build_dir):
                return False
            
            # Install the project
            if not self.install_project(build_dir):
                return False
            
            # Save configuration
            self.save_config(source_dir, build_dir)
            
            # Set up environment
            self.setup_environment()
            
            # Clean up build directory if requested
            if not keep_build:
                shutil.rmtree(build_dir, ignore_errors=True)
                click.echo("Build directory cleaned up")
            
            click.echo("🎉 IOWarp Core installation completed successfully!")
            click.echo(f"Installation prefix: {self.prefix}")
            click.echo(f"To use IOWarp Core, run: source {self.prefix}/bin/iowarp-env.sh")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Installation failed with error: {e}")
            return False


class IOWarpConfig:
    """Configuration manager for IOWarp Core."""
    
    def __init__(self, prefix: Optional[str] = None):
        self.prefix = Path(prefix) if prefix else Path.home() / ".local"
        self.config_file = self.prefix / "etc" / "iowarp" / "config.json"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
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
            
            env_script = self.prefix / "bin" / "iowarp-env.sh"
            click.echo(f"\nEnvironment script: {env_script}")
            if env_script.exists():
                click.echo(f"  Run: source {env_script}")
            else:
                click.echo("  ❌ Environment script not found")


@click.group()
@click.version_option(version="1.0.0", prog_name="iowarp")
def main():
    """IOWarp - Installation tool for IOWarp Core scientific computing platform."""
    pass


@main.command()
@click.option('--prefix', '-p', default=None, 
              help='Installation prefix (default: ~/.local)')
@click.option('--build-type', '-b', default='Release',
              type=click.Choice(['Debug', 'Release', 'RelWithDebInfo']),
              help='Build type (default: Release)')
@click.option('--source-dir', '-s', default=None, type=click.Path(),
              help='Source directory (default: ~/.cache/iowarp-core)')
@click.option('--keep-build', is_flag=True,
              help='Keep build directory after installation')
@click.option('--check-deps-only', is_flag=True,
              help='Only check dependencies, do not install')
def install(prefix: Optional[str], build_type: str, source_dir: Optional[str], 
           keep_build: bool, check_deps_only: bool):
    """Install IOWarp Core components."""
    installer = IOWarpInstaller(prefix=prefix, build_type=build_type)
    
    if check_deps_only:
        deps = installer.check_system_dependencies()
        click.echo("Required dependencies:")
        for dep, available in deps["required"].items():
            status = "✅" if available else "❌"
            click.echo(f"  {status} {dep}")
        
        click.echo("\nOptional dependencies:")
        for dep, available in deps["optional"].items():
            status = "✅" if available else "❌"
            click.echo(f"  {status} {dep}")
        return
    
    source_path = Path(source_dir) if source_dir else None
    success = installer.install(source_dir=source_path, keep_build=keep_build)
    sys.exit(0 if success else 1)


@main.command()
@click.option('--prefix', '-p', default=None,
              help='Installation prefix (default: ~/.local)')
def status(prefix: Optional[str]):
    """Show installation status."""
    config_mgr = IOWarpConfig(prefix=prefix)
    config_mgr.show_status()


@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.option('--prefix', '-p', default=None,
              help='Installation prefix (default: ~/.local)')
def env(prefix: Optional[str]):
    """Show environment variables."""
    config_mgr = IOWarpConfig(prefix=prefix)
    prefix_path = config_mgr.prefix
    
    env_vars = {
        'IOWARP_PREFIX': str(prefix_path),
        'IOWARP_ROOT': str(prefix_path),
        'PATH': f"{prefix_path}/bin:{os.environ.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"{prefix_path}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        'PKG_CONFIG_PATH': f"{prefix_path}/lib/pkgconfig:{os.environ.get('PKG_CONFIG_PATH', '')}",
        'CMAKE_MODULE_PATH': f"{prefix_path}/lib/cmake:{os.environ.get('CMAKE_MODULE_PATH', '')}",
    }
    
    for key, value in env_vars.items():
        click.echo(f"export {key}=\"{value}\"")


@main.command()
@click.option('--component', '-c', default=None,
              type=click.Choice(['runtime', 'cte', 'cae', 'cee', 'transport']),
              help='Run tests for specific component')
@click.option('--list', '-l', 'list_tests', is_flag=True,
              help='List available tests (requires build directory)')
@click.option('--prefix', '-p', default=None,
              help='Installation prefix')
def test(component: Optional[str], list_tests: bool, prefix: Optional[str]):
    """Run IOWarp Core tests."""
    config_mgr = IOWarpConfig(prefix=prefix)
    
    if not config_mgr.is_installed():
        click.echo("❌ IOWarp Core is not installed")
        click.echo("Run 'iowarp install' first to install IOWarp Core")
        sys.exit(1)
    
    config = config_mgr.load_config()
    build_dir = config.get('build_dir')
    
    if not build_dir or not Path(build_dir).exists():
        click.echo("❌ Build directory not found")
        click.echo("Tests require the build directory. Reinstall with --keep-build")
        sys.exit(1)
    
    build_path = Path(build_dir)
    
    if list_tests:
        try:
            result = subprocess.run(
                ["ctest", "--show-only=json-v1"], 
                cwd=build_path, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            test_data = json.loads(result.stdout)
            click.echo("Available tests:")
            for test in test_data.get('tests', []):
                click.echo(f"  - {test.get('name', '')}")
        except Exception as e:
            click.echo(f"❌ Error listing tests: {e}")
        return
    
    # Run tests
    cmd = ["ctest", "-V"]
    
    if component:
        component_patterns = {
            'runtime': 'chimaera.*|hshm.*',
            'cte': 'cte.*|wrp_cte.*',
            'cae': 'cae.*|wrp_cae.*|omni.*',
            'cee': 'cee.*|wrp_cee.*',
            'transport': 'hshm.*',
        }
        
        pattern = component_patterns.get(component.lower())
        if pattern:
            cmd.extend(["-R", pattern])
    
    try:
        result = subprocess.run(cmd, cwd=build_path, check=False)
        if result.returncode == 0:
            click.echo("✅ Tests completed successfully")
        else:
            click.echo("❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()