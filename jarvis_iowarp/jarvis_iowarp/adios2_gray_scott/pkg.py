"""
Gray Scott ADIOS2 Application

3D 7-point stencil code for modeling the diffusion of two substances.
Supports both bare-metal and container deployment modes.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, PsshExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Mkdir, Rm
from jarvis_cd.util.config_parser import JsonFile
import os

# Container build is delegated to jarvis_iowarp.wrp_runtime: every pipeline
# that uses adios2_gray_scott also instantiates wrp_runtime, whose build.sh
# compiles IOWarp with WRP_CORE_ENABLE_GRAY_SCOTT=ON (enabled by the
# 'release-adapter' preset), producing /usr/local/bin/gray-scott in the
# shared build image. So this package contributes no build or deploy
# content of its own; see wrp_runtime/build.sh + Dockerfile.deploy.


class Adios2GrayScott(Application):
    """
    Gray Scott supporting default and container deployment.

    deploy_mode='default': runs gray-scott on the host via MPI.
    deploy_mode='container': builds IOWarp + gray-scott from source,
    copies binaries to a minimal deploy container, runs inside it.
    """

    def _init(self):
        self.adios2_xml_path = f'{self.shared_dir}/adios2.xml'
        self.settings_json_path = f'{self.shared_dir}/settings-files.json'
        self.var_json_path = f'{self.shared_dir}/var.json'
        self.operator_json_path = f'{self.shared_dir}/operator.json'
        self.process = None

    def _configure_menu(self):
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
                'msg': 'Use ADIOS span mode',
                'type': bool,
                'default': False,
            },
            {
                'name': 'adios_memory_selection',
                'msg': 'Use ADIOS memory selection',
                'type': bool,
                'default': False,
            },
            {
                'name': 'mesh_type',
                'msg': 'Mesh type for output',
                'type': str,
                'default': 'image',
            },
            {
                'name': 'engine',
                'msg': 'Engine to be used',
                'choices': ['bp5', 'hermes', 'bp5_derived', 'hermes_derived',
                            'iowarp', 'iowarp_derived', 'sst'],
                'type': str,
                'default': 'bp5',
            },
            {
                'name': 'full_run',
                'msg': 'Will postprocessing be executed?',
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
                'default': '/tmp/benchmark_metadata.db',
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

    # ------------------------------------------------------------------
    # Container build — delegated to wrp_runtime
    # ------------------------------------------------------------------
    # Every pipeline that uses adios2_gray_scott in container mode also
    # spins up wrp_runtime, whose build.sh compiles IOWarp with
    # WRP_CORE_ENABLE_GRAY_SCOTT=ON. That produces /usr/local/bin/gray-scott
    # in the committed build image, which wrp_runtime's Dockerfile.deploy
    # copies into the final deploy image. So no separate build/deploy
    # content is emitted here.
    # _build_phase / _build_deploy_phase inherit the base-class no-op.

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.adios2_xml_path = f'{self.shared_dir}/adios2.xml'
        self.settings_json_path = f'{self.shared_dir}/settings-files.json'
        self.var_json_path = f'{self.shared_dir}/var.json'
        self.operator_json_path = f'{self.shared_dir}/operator.json'

        if self.config['out_file'] is None:
            adios_dir = os.path.join(self.shared_dir, 'gray-scott-output')
            self.config['out_file'] = os.path.join(adios_dir, 'data/out.bp')
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

        engine = self.config['engine'].lower()
        if engine in ['bp5', 'bp5_derived']:
            self.copy_template_file(f'{self.pkg_dir}/config/adios2.xml',
                                    self.adios2_xml_path)
        elif engine == 'sst':
            self.copy_template_file(f'{self.pkg_dir}/config/sst.xml',
                                    self.adios2_xml_path)
        elif engine in ['hermes', 'hermes_derived']:
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
        elif engine in ['iowarp', 'iowarp_derived']:
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
            raise Exception(f'Engine not defined: {engine}')

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        cfg = self.config

        cmd = f'gray-scott {self.settings_json_path}'

        self.process = Exec(cmd, MpiExecInfo(
            nprocs=cfg['nprocs'],
            ppn=cfg['ppn'],
            hostfile=self.hostfile,
            port=self.ssh_port,
            env=self.mod_env,
            exec_async=cfg['run_async'],
            container=self._container_engine,
            container_image=self.deploy_image_name,
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            bind_mounts=self.container_mounts,
        ))
        self.process.run()

    def stop(self):
        if self.config.get('run_async', False) and self.process:
            self.process.wait_all()
        elif self.process:
            self.process.kill_all()

    def clean(self):
        output_file = [self.config['out_file'],
                       self.config['checkpoint_output'],
                       self.config['db_path']]
        Rm(output_file, PsshExecInfo(hostfile=self.hostfile)).run()
