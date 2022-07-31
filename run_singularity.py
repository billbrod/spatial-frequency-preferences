#!/usr/bin/env python3

import argparse
import subprocess
import os
import os.path as op
import json
import re
from glob import glob

# slurm-related paths. change these if your slurm is set up differently or you
# use a different job submission system. see docs
# https://sylabs.io/guides/3.7/user-guide/appendix.html#singularity-s-environment-variables
# for full description of each of these environmental variables
os.environ['SINGULARITY_BINDPATH'] = os.environ.get('SINGULARITY_BINDPATH', '') + ',/opt/slurm,/usr/lib64/libmunge.so.2.0.0,/usr/lib64/libmunge.so.2,/var/run/munge,/etc/passwd'
os.environ['SINGULARITYENV_PREPEND_PATH'] = os.environ.get('SINGULARITYENV_PREPEND_PATH', '') + ':/opt/slurm/bin'
os.environ['SINGULARITY_CONTAINLIBS'] = os.environ.get('SINGULARITY_CONTAINLIBS', '') + ',' + ','.join(glob('/opt/slurm/lib64/libpmi*'))


def check_singularity_envvars():
    """Make sure SINGULARITY_BINDPATH, SINGULARITY_PREPEND_PATH, and SINGULARITY_CONTAINLIBS only contain existing paths
    """
    for env in ['SINGULARITY_BINDPATH', 'SINGULARITYENV_PREPEND_PATH', 'SINGULARITY_CONTAINLIBS']:
        paths = os.environ[env]
        joiner = ',' if env != "SINGULARITYENV_PREPEND_PATH" else ':'
        paths = [p for p in paths.split(joiner) if op.exists(p)]
        os.environ[env] = joiner.join(paths)


def check_bind_paths(volumes):
    """Check that paths we want to bind exist, return only those that do."""
    return [vol for vol in volumes if op.exists(vol.split(':')[0])]


def main(image, args=[], software='singularity', sudo=False):
    """Run sfp singularity container!

    Parameters
    ----------
    image : str
        If running with singularity, the path to the .sif file containing the
        singularity image. If running with docker, name of the docker image.
    args : list, optional
        command to pass to the container. If empty (default), we open up an
        interactive session.
    software : {'singularity', 'docker'}, optional
        Whether to run image with singularity or docker
    sudo : bool, optional
        If True, we run docker with `sudo`. If software=='singularity', we
        ignore this.

    """
    check_singularity_envvars()
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.json')) as f:
        config = json.load(f)
    volumes = [
        f'{op.dirname(op.realpath(__file__))}:/home/sfp_user/spatial-frequency-preferences',
        f'{config["MATLAB_PATH"]}:/home/sfp_user/matlab',
        f'{config["FREESURFER_HOME"]}:/home/sfp_user/freesurfer',
        f'{config["FSLDIR"]}:/home/sfp_user/fsl',
        f'{config["DATA_DIR"]}:{config["DATA_DIR"]}',
        f'{config["WORKING_DIR"]}:{config["WORKING_DIR"]}'
    ]
    volumes = check_bind_paths(volumes)
    # join puts --bind between each of the volumes, we also need it in the
    # beginning
    volumes = '--bind ' + " --bind ".join(volumes)
    # if the user is passing a snakemake command, need to pass
    # --configfile /home/sfp_user/sfp_config.json, since we modify the config
    # file when we source singularity_env.sh
    if args and 'snakemake' == args[0]:
        args = ['snakemake', '--configfile', '/home/sfp_user/sfp_config.json',
                '-d', '/home/sfp_user/spatial-frequency-preferences',
                '-s', '/home/sfp_user/spatial-frequency-preferences/Snakefile', *args[1:]]
    # in this case they passed a string so args[0] contains snakemake and then
    # a bunch of other stuff
    elif args and args[0].startswith('snakemake'):
        args = ['snakemake', '--configfile', '/home/sfp_user/sfp_config.json',
                '-d', '/home/sfp_user/spatial-frequency-preferences',
                '-s', '/home/sfp_user/spatial-frequency-preferences/Snakefile', args[0].replace('snakemake ', ''), *args[1:]]
        # if the user specifies --profile slurm, replace it with the
        # appropriate path. We know it will be in the last one of args and
        # nested below the above elif because if they specified --profile then
        # the whole thing had to be wrapped in quotes, which would lead to this
        # case.
        if '--profile slurm' in args[-1]:
            args[-1] = args[-1].replace('--profile slurm',
                                        '--profile /home/sfp_user/.config/snakemake/slurm')
        # then need to make sure to mount this
        elif '--profile' in args[-1]:
            profile_path = re.findall('--profile (.*?) ', args[-1])[0]
            profile_name = op.split(profile_path)[-1]
            volumes.append(f'{profile_path}:/home/sfp_user/.config/snakemake/{profile_name}')
            args[-1] = args[-1].replace(f'--profile {profile_path}',
                                        f'--profile /home/sfp_user/.config/snakemake/{profile_name}')
    # open up an interactive session if the user hasn't specified an argument,
    # otherwise pass the argument to bash. regardless, make sure we source the
    # env.sh file
    if not args:
        args = ['/bin/bash', '--init-file', '/home/sfp_user/singularity_env.sh']
    else:
        args = ['/bin/bash', '-c',
                # this needs to be done with single quotes on the inside so
                # that's what bash sees, otherwise we run into
                # https://stackoverflow.com/questions/45577411/export-variable-within-bin-bash-c;
                # double-quoted commands get evaluated in the *current* shell,
                # not by /bin/bash -c
                f"'source /home/sfp_user/singularity_env.sh; {' '.join(args)}'"]
    # set these environmental variables, which we use for the jobs submitted to
    # the cluster so they know where to find the container and this script
    env_str = f"--env SFP_PATH={op.dirname(op.realpath(__file__))} --env SINGULARITY_CONTAINER_PATH={image}"
    # the -e flag makes sure we don't pass through any environment variables
    # from the calling shell, while --writable-tmpfs enables us to write to the
    # container's filesystem (necessary because singularity_env.sh makes a
    # temporary config.json file)
    if software == 'singularity':
        exec_str = f'singularity exec -e {env_str} --writable-tmpfs {volumes} {image} {" ".join(args)}'
    elif software == 'docker':
        volumes = volumes.replace('--bind', '--volume')
        exec_str = f'docker run {volumes} -it {image} {" ".join(args)}'
        if sudo:
            exec_str = 'sudo ' + exec_str
    print(exec_str)
    # we use shell=True because we want to carefully control the quotes used
    subprocess.call(exec_str, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run billbrod/sfp container. This is a wrapper, which binds the appropriate"
                     " paths and sources singularity_env.sh, setting up some environmental variables.")
    )
    parser.add_argument('image',
                        help=('If running with singularity, the path to the '
                              '.sif file containing the singularity image. '
                              'If running with docker, name of the docker image.'))
    parser.add_argument('--software', default='singularity', choices=['singularity', 'docker'],
                        help="Whether to run this with singularity or docker")
    parser.add_argument('--sudo', '-s', action='store_true',
                        help="Whether to run docker with sudo or not. Ignored if software==singularity")
    parser.add_argument("args", nargs='*',
                        help=("Command to pass to the container. If empty, we open up an interactive session."
                              " If it contains flags, surround with SINGLE QUOTES (not double)."))
    args = vars(parser.parse_args())
    main(**args)
