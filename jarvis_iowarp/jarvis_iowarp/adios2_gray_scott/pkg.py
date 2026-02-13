"""
This module provides classes and methods to launch the Gray Scott application.
Gray Scott is a 3D 7-point stencil code for modeling the diffusion of two
substances.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, PsshExecInfo
from jarvis_cd.shell.process import Mkdir, Rm
from jarvis_cd.util.config_parser import JsonFile
import pathlib
import os


class Adios2GrayScott(Application):
    """
    This class provides methods to launch the GrayScott application.
    """
    def _init(self):
        """
        Initialize paths
        """
        self.adios2_xml_path = f'{self.shared_dir}/adios2.xml'
        self.settings_json_path = f'{self.shared_dir}/settings-files.json'
        self.var_json_path = f'{self.shared_dir}/var.json'
        self.operator_json_path = f'{self.shared_dir}/operator.json'
        self.process = None  # Store process handle for async execution

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.
        For thorough documentation of these parameters, view:
        https://github.com/scs-lab/jarvis-util/wiki/3.-Argument-Parsing

        :return: List(dict)
        """
        return [
            {
                'name': 'nprocs',
                'msg': 'Number of processes to spawn',
                'type': int,
                'default': 4,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 16,
            },
            {
                'name': 'L',
                'msg': 'Grid size of cube',
                'type': int,
                'default': 32,
            },
            {
                'name': 'Du',
                'msg': 'Diffusion rate of substance U',
                'type': float,
                'default': .2,
            },
            {
                'name': 'Dv',
                'msg': 'Diffusion rate of substance V',
                'type': float,
                'default': .1,
            },
            {
                'name': 'F',
                'msg': 'Feed rate of U',
                'type': float,
                'default': .01,
            },
            {
                'name': 'k',
                'msg': 'Kill rate of V',
                'type': float,
                'default': .05,
            },
            {
                'name': 'dt',
                'msg': 'Timestep',
                'type': float,
                'default': 2.0,
            },
            {
                'name': 'steps',
                'msg': 'Total number of steps to simulate',
                'type': int,
                'default': 100,
            },
            {
                'name': 'plotgap',
                'msg': 'Number of steps between output',
                'type': float,
                'default': 10,
            },
            {
                'name': 'noise',
                'msg': 'Amount of noise',
                'type': float,
                'default': .01,
            },
            {
                'name': 'out_file',
                'msg': 'Absolute path to output file',
                'type': str,
                'default': None,
            },
            {
                'name': 'checkpoint',
                'msg': 'Perform checkpoints',
                'type': bool,
                'default': True,
            },
            {
                'name': 'checkpoint_freq',
                'msg': 'Frequency of the checkpoints',
                'type': int,
                'default': 70,
            },
            {
                'name': 'checkpoint_output',
                'msg': 'Output location of the checkpoint',
                'type': str,
                'default': 'ckpt.bp',
            },
            {
                'name': 'restart',
                'msg': 'Perform restarts',
                'type': bool,
                'default': False,
            },
            {
                'name': 'restart_input',
                'msg': 'Input for the restart',
                'type': str,
                'default': 'ckpt.bp',
            },
            {
                'name': 'adios_span',
                'msg': '???',
                'type': bool,
                'default': False,
            },
            {
                'name': 'adios_memory_selection',
                'msg': '???',
                'type': bool,
                'default': False,
            },
            {
                'name': 'mesh_type',
                'msg': '???',
                'type': str,
                'default': 'image',
            },
            {
                'name': 'engine',
                'msg': 'Engine to be used',
                'choices': ['bp5', 'hermes', 'bp5_derived', 'hermes_derived', 'iowarp', 'iowarp_derived', 'sst'],
                'type': str,
                'default': 'bp5',
            },
            {
                'name': 'full_run',
                'msg': 'Whill postprocessing be executed?',
                'type': bool,
                'default': True,
            },
            {
                'name': 'limit',
                'msg': 'Limit the value of data to track',
                'type': int,
                'default': 0,
            },
            {
                'name': 'db_path',
                'msg': 'Path where the DB will be stored',
                'type': str,
                'default': 'benchmark_metadata.db',
            },
            {
                'name': 'Execution_order',
                'msg': 'Path where the bp5 will be stored',
                'type': str,
                'default': '1',
            },
            {
                'name': 'run_async',
                'msg': 'Run in background for parallel execution with consumer',
                'type': bool,
                'default': False,
            },

        ]

    # jarvis pkg config adios2_gray_scott ppn=20 full_run=true engine=hermes db_path=/mnt/nvme/jcernudagarcia/metadata.db out_file=gs.bp nprocs=1

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        if self.config['out_file'] is None:
            adios_dir = os.path.join(self.shared_dir, 'gray-scott-output')
            self.config['out_file'] = os.path.join(adios_dir,
                                                 'data/out.bp')
            Mkdir(adios_dir, PsshExecInfo(hostfile=self.hostfile,
                                          env=self.env)).run()
        settings_json = {
            'L': self.config['L'],
            'Du': self.config['Du'],
            'Dv': self.config['Dv'],
            'F': self.config['F'],
            'k': self.config['k'],
            'dt': self.config['dt'],
            'plotgap': self.config['plotgap'],
            'steps': self.config['steps'],
            'noise': self.config['noise'],
            'output': self.config['out_file'],
            'checkpoint': self.config['checkpoint'],
            'checkpoint_freq': self.config['checkpoint_freq'],
            'checkpoint_output': self.config['checkpoint_output'],
            'restart': self.config['restart'],
            'restart_input': self.config['restart_input'],
            'adios_span': self.config['adios_span'],
            'adios_memory_selection': self.config['adios_memory_selection'],
            'mesh_type': self.config['mesh_type'],
            'adios_config': f'{self.adios2_xml_path}'
        }
        output_dir = os.path.dirname(self.config['out_file'])
        db_dir = os.path.dirname(self.config['db_path'])
        Mkdir([output_dir, db_dir], PsshExecInfo(hostfile=self.hostfile,
                                       env=self.env)).run()

        JsonFile(self.settings_json_path).save(settings_json)
        print(f"Using engine {self.config['engine']}")
        if self.config['engine'].lower() in ['bp5', 'bp5_derived']:
            self.copy_template_file(f'{self.pkg_dir}/config/adios2.xml',
                                self.adios2_xml_path)
        elif self.config['engine'].lower() == 'sst':
            self.copy_template_file(f'{self.pkg_dir}/config/sst.xml',
                                self.adios2_xml_path)
        elif self.config['engine'].lower() in ['hermes', 'hermes_derived']:
            self.copy_template_file(f'{self.pkg_dir}/config/hermes.xml',
                                    self.adios2_xml_path,
                                    replacements={
                                        'PPN': self.config['ppn'],
                                        'VARFILE': self.var_json_path,
                                        'OPFILE': self.operator_json_path,
                                        'DBFILE': self.config['db_path'],
                                        'Order': self.config['Execution_order'],
                                    })
            self.copy_template_file(f'{self.pkg_dir}/config/var.yaml',
                                    self.var_json_path)
            self.copy_template_file(f'{self.pkg_dir}/config/operator.yaml',
                                    self.operator_json_path)
        elif self.config['engine'].lower() in ['iowarp', 'iowarp_derived']:
            self.copy_template_file(f'{self.pkg_dir}/config/iowarp.xml',
                                    self.adios2_xml_path,
                                    replacements={
                                        'PPN': self.config['ppn'],
                                        'VARFILE': self.var_json_path,
                                        'OPFILE': self.operator_json_path,
                                        'DBFILE': self.config['db_path'],
                                        'Order': self.config['Execution_order'],
                                    })
            self.copy_template_file(f'{self.pkg_dir}/config/var.yaml',
                                    self.var_json_path)
            self.copy_template_file(f'{self.pkg_dir}/config/operator.yaml',
                                    self.operator_json_path)
        else:
            raise Exception('Engine not defined')

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        # print(self.env['HERMES_CLIENT_CONF'])
        if self.config['engine'].lower() in ['bp5_derived', 'hermes_derived', 'iowarp_derived']:
            derived = 1
            self.process = Exec(f'gray-scott {self.settings_json_path} {derived}',
                 MpiExecInfo(nprocs=self.config['nprocs'],
                             ppn=self.config['ppn'],
                             hostfile=self.hostfile,
                             env=self.mod_env,
                             exec_async=self.config['run_async']
                             ))
            self.process.run()
        elif self.config['engine'].lower() in ['hermes', 'bp5', 'iowarp', 'sst']:

            derived = 0
            self.process = Exec(f'gray-scott {self.settings_json_path} {derived}',
                 MpiExecInfo(nprocs=self.config['nprocs'],
                             ppn=self.config['ppn'],
                             hostfile=self.hostfile,
                             env=self.mod_env,
                             exec_async=self.config['run_async']))
            self.process.run()


    def wait(self):
        """
        Wait for async process to complete.

        :return: None
        """
        if self.process:
            self.process.wait_all()

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        # If running async, wait for completion instead of killing
        if self.config.get('run_async', False) and self.process:
            print("Waiting for async gray-scott producer to complete...")
            self.process.wait_all()
        elif self.process:
            self.process.kill_all()
        pass

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        output_file = [self.config['out_file'],
                       self.config['checkpoint_output'],
                       self.config['db_path']
                       ]

        print(f'Removing {output_file}')
        Rm(output_file, PsshExecInfo(hostfile=self.hostfile)).run()
