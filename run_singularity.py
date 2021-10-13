#!/usr/bin/env python3

import argparse
import subprocess
import os.path as op
import yaml
import re


def main(path, args=[]):
    """
    """
    with open(op.join(op.dirname(op.realpath(__file__)), 'config.yml')) as f:
        config = yaml.safe_load(f)
    volumes = [
        '.:/home/sfp_user/spatial-frequency-preferences',
        f'{config["MATLAB_PATH"]}:/home/sfp_user/matlab',
        f'{config["FREESURFER_HOME"]}:/home/sfp_user/freesurfer',
        f'{config["FSLDIR"]}:/home/sfp_user/fsl',
        f'{config["DATA_DIR"]}:/home/sfp_user/sfp_data',
    ]
    # join puts --bind between each of the volumes, we also need it in the
    # beginning
    volumes = '--bind ' + " --bind ".join(volumes)
    # make sure we use the image-internal data directory
    if any([config['DATA_DIR'] in a for a in args]):
        args = [a.replace(config['DATA_DIR'], '/home/sfp_user/sfp_data') for a in args]
    # if the user is passing a snakemake command, need to pass
    # --configfile /home/sfp_user/sfp_config.yml, since we modify the config
    # file when we source singularity_env.sh
    if args and 'snakemake' == args[0]:
        args = ['snakemake', '--configfile', '/home/sfp_user/sfp_config.yml', *args[1:]]
    # in this case they passed a string so args[0] contains snakemake and then
    # a bunch of other stuff
    elif args and args[0].startswith('snakemake'):
        args = ['snakemake', '--configfile', '/home/sfp_user/sfp_config.yml',
                args[0].replace('snakemake ', ''), *args[1:]]
    # if the user specifies --profile slurm, replace it with the appropriate
    # path. We know it will be in the last one of args and nested below the
    # above elif because if they specified --profile then the whole thing had
    # to be wrapped in double quotes, which would lead to this case.
        if '--profile slurm' in args[-1]:
            args[-1] = args[-1].replace('--profile slurm',
                                        '--profile /home/sfp_user/.config/snakemake/slurm --cluster-config cluster.json')
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
    # the -e flag makes sure we don't pass through any environment variables
    # from the calling shell, while --writable-tmpfs enables us to write to the
    # container's filesystem (necessary because singularity_env.sh makes a
    # temporary config.yml file)
    exec_str = f'singularity exec -e --writable-tmpfs {volumes} {path} {" ".join(args)}'
    # we use shell=True because we want to carefully control the quotes used
    print(exec_str)
    subprocess.call(exec_str, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run billbrod/sfp container. This is a wrapper, which binds the appropriate"
                     " paths and sources singularity_env.sh, setting up some environmental variables.")
    )
    parser.add_argument('path', help='Path to the .sif image.')
    parser.add_argument("args", nargs='*',
                        help="Command to pass to the container. If empty, we open up an interactive session.")
    args = vars(parser.parse_args())
    main(**args)
