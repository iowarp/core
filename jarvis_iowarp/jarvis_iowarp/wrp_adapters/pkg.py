"""
This module provides classes and methods to inject WRP (IoWarp) CTE adapters.
The WrpAdapters interceptor enables interception of various I/O APIs (POSIX,
MPI-IO, STDIO, HDF5 VFD, NVIDIA GDS) and routes them to the Content Transfer
Engine for intelligent data placement and transfer.
"""
from jarvis_cd.core.pkg import Interceptor
from jarvis_cd.util import SizeType
import pathlib
import os
import yaml


class WrpAdapters(Interceptor):
    """
    WRP CTE Adapters Interceptor.

    This interceptor enables I/O interception for various APIs by setting up
    LD_PRELOAD with the appropriate WRP CTE adapter libraries. It supports:
    - POSIX I/O (read, write, open, close, etc.)
    - MPI-IO (MPI_File_* operations)
    - STDIO (fread, fwrite, fopen, etc.)
    - HDF5 VFD (Virtual File Driver for HDF5)
    - NVIDIA GDS (GPUDirect Storage)
    """

    def _init(self):
        """
        Initialize the interceptor.

        This method is called during package instantiation.
        No configuration is available yet at this stage.
        """
        self.cae_config_path = None

    def _configure_menu(self):
        """
        Define configuration options for the WRP adapters interceptor.

        Returns:
            List[Dict]: Configuration parameters for adapter selection.
        """
        return [
            {
                'name': 'posix',
                'msg': 'Intercept POSIX I/O operations',
                'type': bool,
                'default': False,
                'help': 'Intercepts read, write, open, close, lseek, etc.'
            },
            {
                'name': 'mpiio',
                'msg': 'Intercept MPI-IO operations',
                'type': bool,
                'default': False,
                'help': 'Intercepts MPI_File_* operations'
            },
            {
                'name': 'stdio',
                'msg': 'Intercept STDIO operations',
                'type': bool,
                'default': False,
                'help': 'Intercepts fread, fwrite, fopen, fclose, etc.'
            },
            {
                'name': 'vfd',
                'msg': 'Intercept HDF5 I/O via VFD',
                'type': bool,
                'default': False,
                'help': 'Enables HDF5 Virtual File Driver for CTE'
            },
            {
                'name': 'nvidia_gds',
                'msg': 'Intercept NVIDIA GDS I/O',
                'type': bool,
                'default': False,
                'help': 'Intercepts NVIDIA GPUDirect Storage operations'
            },
            {
                'name': 'adios2',
                'msg': 'Enable ADIOS2 IOWarp engine plugin',
                'type': bool,
                'default': False,
                'help': 'Sets ADIOS2_PLUGIN_PATH so ADIOS2 discovers the iowarp_engine plugin'
            },
            {
                'name': 'include',
                'msg': 'Path patterns to include for interception (list)',
                'type': list,
                'default': ['.*test_hermes.*'],
                'help': 'List of regex patterns for paths to intercept. Supports env var expansion ($HOME, ${VAR}, ~). Example: [".*test_hermes.*", "/data/.*", "$HOME/scratch/.*"]'
            },
            {
                'name': 'exclude',
                'msg': 'Path patterns to exclude from interception (list)',
                'type': list,
                'default': [],
                'help': 'List of regex patterns for paths to exclude from interception. Takes precedence based on specificity (longer paths checked first). Example: ["/tmp/.*", "/dev/.*"]'
            },
            {
                'name': 'adapter_page_size',
                'msg': 'Adapter page size for I/O operations',
                'type': str,
                'default': '1M',
                'help': 'Page size for adapter operations. Supports SizeType format (e.g., "4K", "1M", "4096")'
            },
        ]

    def _configure(self, **kwargs):
        """
        Configure the WRP adapters interceptor.

        This method finds the adapter libraries, stores their paths
        in the environment, and generates the CAE configuration file.

        Args:
            **kwargs: Configuration parameters (automatically updated to self.config)

        Raises:
            Exception: If no adapter is selected or if a selected adapter library
                      cannot be found.
        """
        has_one = False

        if self.config['posix']:
            posix_lib = self.find_library('wrp_cte_posix')
            if posix_lib is None:
                raise Exception('Could not find wrp_cte_posix library')
            self.env['WRP_CTE_POSIX'] = posix_lib
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(posix_lib).parent.parent)
            self.log(f'Found libwrp_cte_posix.so at {posix_lib}')
            has_one = True

        if self.config['mpiio']:
            mpiio_lib = self.find_library('wrp_cte_mpiio')
            if mpiio_lib is None:
                raise Exception('Could not find wrp_cte_mpiio library')
            self.env['WRP_CTE_MPIIO'] = mpiio_lib
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(mpiio_lib).parent.parent)
            self.log(f'Found libwrp_cte_mpiio.so at {mpiio_lib}')
            has_one = True

        if self.config['stdio']:
            stdio_lib = self.find_library('wrp_cte_stdio')
            if stdio_lib is None:
                raise Exception('Could not find wrp_cte_stdio library')
            self.env['WRP_CTE_STDIO'] = stdio_lib
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(stdio_lib).parent.parent)
            self.log(f'Found libwrp_cte_stdio.so at {stdio_lib}')
            has_one = True

        if self.config['vfd']:
            vfd_lib = self.find_library('wrp_cte_vfd')
            if vfd_lib is None:
                raise Exception('Could not find wrp_cte_vfd library')
            self.env['WRP_CTE_VFD'] = vfd_lib
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(vfd_lib).parent.parent)
            self.log(f'Found libwrp_cte_vfd.so at {vfd_lib}')
            has_one = True

        if self.config['nvidia_gds']:
            nvidia_gds_lib = self.find_library('wrp_cte_nvidia_gds')
            if nvidia_gds_lib is None:
                raise Exception('Could not find wrp_cte_nvidia_gds library')
            self.env['WRP_CTE_NVIDIA_GDS'] = nvidia_gds_lib
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(nvidia_gds_lib).parent.parent)
            self.log(f'Found libwrp_cte_nvidia_gds.so at {nvidia_gds_lib}')
            has_one = True

        if self.config['adios2']:
            adios2_lib = self.find_library('iowarp_engine')
            if adios2_lib is None:
                raise Exception('Could not find iowarp_engine library')
            self.env['ADIOS2_PLUGIN_PATH'] = str(pathlib.Path(adios2_lib).parent)
            self.env['WRP_CTE_ROOT'] = str(pathlib.Path(adios2_lib).parent.parent)
            self.log(f'Found libiowarp_engine.so at {adios2_lib}')
            has_one = True

        if not has_one:
            raise Exception('No WRP CTE adapter selected. Please enable at least one adapter (posix, mpiio, stdio, vfd, nvidia_gds, or adios2).')

        # Generate CAE configuration
        self._generate_cae_config()

    def modify_env(self):
        """
        Modify the environment to enable WRP CTE adapter interception.

        This method is called automatically by Jarvis during pipeline start,
        just before the target package's start() method. It modifies the shared
        mod_env to add adapter libraries to LD_PRELOAD.

        The mod_env is shared between the interceptor and the target package,
        so changes made here directly affect the package's execution environment.
        """
        # Set CAE configuration environment variable
        if self.cae_config_path and os.path.exists(self.cae_config_path):
            self.setenv('WRP_CAE_CONF', self.cae_config_path)
            self.log(f"Set WRP_CAE_CONF to {self.cae_config_path}")

        # Add POSIX adapter to LD_PRELOAD
        if self.config['posix']:
            self.prepend_env('LD_PRELOAD', self.env['WRP_CTE_POSIX'])
            self.log(f"Added POSIX adapter to LD_PRELOAD")

        # Add MPI-IO adapter to LD_PRELOAD
        if self.config['mpiio']:
            self.prepend_env('LD_PRELOAD', self.env['WRP_CTE_MPIIO'])
            self.log(f"Added MPI-IO adapter to LD_PRELOAD")

        # Add STDIO adapter to LD_PRELOAD
        if self.config['stdio']:
            self.prepend_env('LD_PRELOAD', self.env['WRP_CTE_STDIO'])
            self.log(f"Added STDIO adapter to LD_PRELOAD")

        # Configure HDF5 VFD (uses plugin path instead of LD_PRELOAD)
        if self.config['vfd']:
            plugin_path_parent = str(pathlib.Path(self.env['WRP_CTE_VFD']).parent)
            self.setenv('HDF5_PLUGIN_PATH', plugin_path_parent)
            self.setenv('HDF5_DRIVER', 'wrp_cte_vfd')
            self.log(f"Configured HDF5 VFD with plugin path: {plugin_path_parent}")

        # Add NVIDIA GDS adapter to LD_PRELOAD
        if self.config['nvidia_gds']:
            self.prepend_env('LD_PRELOAD', self.env['WRP_CTE_NVIDIA_GDS'])
            self.log(f"Added NVIDIA GDS adapter to LD_PRELOAD")

        # Configure ADIOS2 plugin path
        if self.config['adios2']:
            self.setenv('ADIOS2_PLUGIN_PATH', self.env['ADIOS2_PLUGIN_PATH'])
            self.log(f"Set ADIOS2_PLUGIN_PATH to {self.env['ADIOS2_PLUGIN_PATH']}")

    def _generate_cae_config(self):
        """
        Generate the Content Adapter Engine (CAE) configuration file.

        This creates a YAML configuration file that specifies:
        - include: List of regex patterns for paths to include
        - exclude: List of regex patterns for paths to exclude
        - adapter_page_size: Page size for adapter I/O operations

        Paths are checked in order of specificity (longest path first).
        If a path matches an include pattern, it's intercepted.
        If it matches an exclude pattern, it's not intercepted.
        If no patterns match, the path is excluded by default.

        The configuration file is saved to the shared directory and its path
        is stored in self.cae_config_path for use in modify_env().
        """
        # Parse adapter page size using SizeType
        page_size_bytes = SizeType(self.config['adapter_page_size']).bytes

        # Get include paths from configuration and expand environment variables
        include_patterns = self.config.get('include', ['.*test_hermes.*'])
        expanded_includes = []
        for pattern in include_patterns:
            # Expand environment variables (e.g., $HOME, ${SCRATCH_DIR})
            expanded_pattern = os.path.expandvars(pattern)
            # Also expand user home directory (e.g., ~)
            expanded_pattern = os.path.expanduser(expanded_pattern)
            expanded_includes.append(expanded_pattern)

            # Log if pattern was expanded
            if expanded_pattern != pattern:
                self.log(f"Expanded include pattern: {pattern} -> {expanded_pattern}")

        # Get exclude paths from configuration and expand environment variables
        exclude_patterns = self.config.get('exclude', [])
        expanded_excludes = []
        for pattern in exclude_patterns:
            # Expand environment variables
            expanded_pattern = os.path.expandvars(pattern)
            # Also expand user home directory
            expanded_pattern = os.path.expanduser(expanded_pattern)
            expanded_excludes.append(expanded_pattern)

            # Log if pattern was expanded
            if expanded_pattern != pattern:
                self.log(f"Expanded exclude pattern: {pattern} -> {expanded_pattern}")

        # Build CAE configuration
        cae_config = {
            'include': expanded_includes,
            'exclude': expanded_excludes,
            'adapter_page_size': page_size_bytes
        }

        # Save configuration to shared directory
        self.cae_config_path = os.path.join(self.shared_dir, 'cae_config.yaml')

        try:
            with open(self.cae_config_path, 'w') as f:
                yaml.dump(cae_config, f, default_flow_style=False, indent=2)

            # Set environment variable immediately after creating config file
            self.env['WRP_CAE_CONF'] = self.cae_config_path

            self.log(f"CAE configuration written to: {self.cae_config_path}")
            self.log(f"Set WRP_CAE_CONF={self.cae_config_path}")

            # Log configuration details
            if expanded_includes:
                self.log(f"Include patterns ({len(expanded_includes)}): {', '.join(expanded_includes)}")
            else:
                self.log("No include patterns specified - using default: .*test_hermes.*")

            if expanded_excludes:
                self.log(f"Exclude patterns ({len(expanded_excludes)}): {', '.join(expanded_excludes)}")
            else:
                self.log("No exclude patterns specified")

            self.log(f"Adapter page size: {SizeType(page_size_bytes).to_human_readable()}")
            self.log("Path matching: Most specific pattern wins, default is exclude")

        except Exception as e:
            self.log(f"Error writing CAE configuration: {e}")
            raise

    def clean(self):
        """
        Clean up interceptor data.

        WRP adapters typically don't create persistent data that needs cleanup.
        The CAE configuration file is left in place as it may be useful for
        debugging or reuse.
        """
        pass
