"""
IOWarp FUSE adapter service.

Mounts the CTE-backed virtual filesystem at a configured path by launching
the `wrp_cte_fuse` binary (from context-transfer-engine/adapter/libfuse).
The binary is built when wrp_runtime is compiled with a CMake preset that
has WRP_CTE_ENABLE_FUSE_ADAPTER=ON (default for release-adapter).

Applications running in the same pipeline can then read/write normally to
paths under the mountpoint, and the FUSE adapter transparently converts
POSIX I/O into CTE PutBlob/GetBlob operations against Chimaera.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Mkdir, Rm
import os
import subprocess


class WrpCteLibfuse(Service):
    """
    IOWarp FUSE adapter — mounts the CTE filesystem at a configured path.

    deploy_mode='default':   mounts on the host (requires libfuse3 + /dev/fuse).
    deploy_mode='container': mounts inside the pipeline's deploy container.
                             docker/podman: pipeline must grant the
                             container CAP_SYS_ADMIN and bind /dev/fuse
                             via container_extensions.
                             apptainer (unprivileged): the kernel forces
                             nodev on user --bind mounts, so a FUSE
                             program inside the instance gets EACCES on
                             open(/dev/fuse) even with --fakeroot +
                             --add-caps SYS_ADMIN. start() therefore uses
                             apptainer's --fusemount helper, which opens
                             /dev/fuse on the host side and hands the fd
                             into the namespace. No container_caps /
                             container_binds entries are needed for FUSE.
    """

    def _init(self):
        # Absolute path — jarvis runs the command via SSH/docker-exec with
        # a minimal shell environment that may not include /usr/local/bin.
        self.binary = '/usr/local/bin/wrp_cte_fuse'

    def _configure_menu(self):
        return [
            {
                'name': 'mountpoint',
                'msg': ('Absolute path to mount the CTE filesystem. The '
                        'directory is created if it does not exist and '
                        'unmounted on stop.'),
                'type': str,
                'default': '/mnt/wrp_cte',
            },
            {
                'name': 'log_level',
                'msg': 'HSHM log level for the FUSE daemon (debug/info/warn/error)',
                'type': str,
                'choices': ['debug', 'info', 'warning', 'error'],
                'default': 'info',
            },
            {
                'name': 'extra_fuse_args',
                'msg': 'Extra CLI flags forwarded to wrp_cte_fuse (passes through to libfuse).',
                'type': str,
                'default': '',
            },
            {
                'name': 'sleep',
                'msg': 'Seconds to wait after mount for the FUSE client handshake to settle.',
                'type': int,
                'default': 1,
            },
        ]

    # No container build/deploy phase — the binary ships inside the
    # wrp_runtime build image when WRP_CTE_ENABLE_FUSE_ADAPTER=ON, and
    # wrp_runtime's Dockerfile.deploy copies /usr/local/bin into the
    # final pipeline image.

    def _configure(self, **kwargs):
        super()._configure(**kwargs)
        self.setenv('HSHM_LOG_LEVEL', self.config['log_level'])
        # wrp_cte_fuse's main() calls CHIMAERA_INIT(kClient, true), where
        # the second arg is default_with_runtime — it makes the FUSE
        # daemon also try to start a runtime (server) on port 9413, which
        # then conflicts with the already-running wrp_runtime service.
        # Forcing CHI_WITH_RUNTIME=0 keeps the daemon strictly client-mode.
        self.setenv('CHI_WITH_RUNTIME', '0')
        # NOTE: the mountpoint is created in start() — at _configure time
        # the pipeline's deploy container may not exist yet, so an SSH
        # Mkdir would land on the host filesystem instead.

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _exec_in_container(self, shell_cmd, env=None):
        """
        Run ``shell_cmd`` inside the pipeline's deploy container.

        Engine dispatch:
          - docker / podman: ``<engine> exec <pipeline>_container ...``.
          - apptainer: ``apptainer exec instance://<pipeline>`` — we rely
            on the instance being started in start() (apptainer has no
            long-running daemonic container otherwise, so a plain
            ``apptainer exec <sif>`` creates a fresh ephemeral process
            group each call and a FUSE mount in one wouldn't persist to
            the next).
          - bare-metal / engine == 'none': run on host.

        Returns the container exec return code.
        """
        engine = self._container_engine
        if not engine or engine == 'none':
            return subprocess.call(['bash', '-c', shell_cmd], env=env)
        image = self.deploy_image_name()
        if engine == 'apptainer':
            # Route via the long-lived instance we started in start().
            # --cleanenv + explicit --env so only our vars reach the
            # process (apptainer leaks host $PATH etc. otherwise).
            cmd = ['apptainer', 'exec', '--cleanenv']
            for k, v in (env or {}).items():
                cmd.extend(['--env', f'{k}={v}'])
            cmd.extend([f'instance://{image}', 'bash', '-c', shell_cmd])
            self.log(f"apptainer exec instance://{image}: {shell_cmd}")
            return subprocess.call(cmd)
        container = f'{image}_container'
        cmd = [engine, 'exec']
        for k, v in (env or {}).items():
            cmd.extend(['-e', f'{k}={v}'])
        cmd.extend([container, 'bash', '-c', shell_cmd])
        self.log(f"exec in {container}: {shell_cmd}")
        return subprocess.call(cmd)

    def start(self):
        mp = self.config['mountpoint']
        extra = self.config.get('extra_fuse_args', '').strip()

        # The deploy container only exists once the pipeline is up, so
        # we create the mountpoint here rather than in _configure().
        if self._exec_in_container(f'mkdir -p {mp}') != 0:
            raise RuntimeError(f'Failed to create mountpoint {mp}')

        engine = self._container_engine
        if engine == 'apptainer':
            # Unprivileged Apptainer forces nodev on every user --bind,
            # including --bind /dev/fuse, so a FUSE program inside the
            # instance gets EACCES when it open()s /dev/fuse — even with
            # --fakeroot and --add-caps SYS_ADMIN. Apptainer's own
            # --fusemount sidesteps this: its setuid helper opens
            # /dev/fuse on the host and passes the fd into the namespace.
            # type=container-daemon = run the FUSE binary inside the
            # container, detached; the exec returns once the mount is
            # live, and the daemon stays up for the instance lifetime.
            image = self.deploy_image_name()
            fuse_cmd = self.binary + ((' ' + extra) if extra else '')
            fuse_spec = f'container-daemon:{fuse_cmd} {mp}'
            cmd_list = ['apptainer', 'exec', '--cleanenv']
            for k, v in (self.mod_env or {}).items():
                cmd_list.extend(['--env', f'{k}={v}'])
            cmd_list.extend(['--fusemount', fuse_spec,
                             f'instance://{image}', '/bin/true'])
            self.log(f"Mounting IOWarp CTE FUSE at {mp} via "
                     f"apptainer --fusemount: {' '.join(cmd_list)}")
            rc = subprocess.call(cmd_list)
        else:
            cmd = f'{self.binary} {mp}'
            if extra:
                cmd = f'{cmd} {extra}'
            self.log(f"Mounting IOWarp CTE FUSE at {mp}: {cmd}")
            rc = self._exec_in_container(cmd, env=self.mod_env)
        if rc != 0:
            raise RuntimeError(
                f'wrp_cte_fuse failed to mount at {mp} (exit={rc})')

        # Let the FUSE <-> chimaera handshake settle before downstream
        # packages (e.g., IOR) start writing through the mount.
        import time
        time.sleep(self.config.get('sleep', 1))

    def stop(self):
        mp = self.config['mountpoint']
        if self._container_engine == 'apptainer':
            # The mount is owned by Apptainer's fusemount helper (started
            # via --fusemount in start()); fusermount3 from inside the
            # instance can't unmount it ("Invalid argument"). The daemon
            # and mount are torn down when the instance stops, so this
            # path is a no-op.
            self.log(f"Skipping in-container unmount of {mp}; apptainer "
                     "instance teardown will release the fusemount.")
            return
        self.log(f"Unmounting {mp}")
        # fusermount3 ships with libfuse3 and is the canonical unmount
        # for user-mounted FUSE filesystems.
        self._exec_in_container(f'fusermount3 -u {mp} || true')

    def clean(self):
        # Don't delete the mountpoint directory — callers may have seeded
        # it (e.g., bound to a specific location). Just make sure nothing
        # is still mounted.
        self.stop()
