"""Command line ingest: make an entarchy, and put experiments into it.

    python -m vxentarchy create /data/entarchy
    python -m vxentarchy add /data/entarchy /raw/cmn
    python -m vxentarchy scan /raw/cmn

`add` is the whole of an experiment folder - `<experiment>/<animal>/<recording>`
- and may be run again whenever the folder has grown, since animals and
recordings already in the entarchy are skipped.

How frames were timed cannot be detected: each recording folder says so in its
metadata.yaml (`frame_sync_signal_type`, `frame_sync_signal`,
`frame_sync_signal_timing`). A folder with imaging output and no such entry is
reported, listed at the end, and left out until it has one. The choice is
written onto the Experiment and onto each Imaging entity, so the entarchy
records how it read its own data.
"""
from __future__ import annotations

import argparse
import os
import sys

from .schema import ImagingSpec, VxEntarchy, lacks_frame_timing, scan_experiment

CONFIG_NAME = 'entarchy.yaml'


def _add_backend_options(parser):
    group = parser.add_argument_group(
        'backend', 'only used when creating; without any of these the entarchy '
                   'is an SQLite database, entarchy.db, in the directory')
    group.add_argument('--backend', choices=['sqlite', 'mysql'], default='sqlite')
    group.add_argument('--dbname', default=None,
                       help='database file for sqlite (default entarchy.db), '
                            'schema name for mysql')
    group.add_argument('--dbhost', default=None, help='mysql only')
    group.add_argument('--dbuser', default=None, help='mysql only')


def _backend_config(args) -> tuple[str, dict | None]:
    """The backend name and its configuration, as create() takes them.

    A mysql password is never taken as an argument - it would end up in the
    shell history - and never written to entarchy.yaml. The backend reads
    ENTARCHY_DB_PASSWORD or asks.
    """
    if args.backend == 'sqlite':
        if args.dbhost or args.dbuser:
            raise SystemExit('--dbhost and --dbuser are for --backend mysql.')
        # entarchy's own default, entarchy.db, unless a name is given
        return 'SQLiteBackend', {'dbname': args.dbname} if args.dbname else None

    missing = [flag for flag, value in (('--dbname', args.dbname), ('--dbhost', args.dbhost),
                                        ('--dbuser', args.dbuser)) if not value]
    if missing:
        raise SystemExit(f'--backend mysql needs {", ".join(missing)}.')

    return 'MySQLBackend', {'dbname': args.dbname, 'dbhost': args.dbhost,
                            'dbuser': args.dbuser}


def _imaging_from_args(args):
    """What `add_experiment(imaging=...)` should be given."""
    if args.imaging == 'none':
        return None
    if args.imaging == 'auto':
        return 'auto'

    return ImagingSpec(source=args.imaging)


def _contents(folder: str, limit: int = None) -> list:
    if not os.path.isdir(folder):
        raise SystemExit(f'No experiment folder {folder}')

    contents = scan_experiment(folder)
    if limit is not None:
        contents = [(animal_path, recordings[:limit])
                    for animal_path, recordings in contents]

    if len(contents) == 0:
        raise SystemExit(f'{folder} holds no animal folder with a vxpy recording '
                         f'in it. An experiment folder is '
                         f'<experiment>/<animal>/<recording>.')

    return contents


def _print_contents(folder: str, contents: list, check_timing: bool = True) -> None:
    print(f'\n{folder}')
    for animal_path, recordings in contents:
        print(f'  {os.path.basename(animal_path):<46} {len(recordings)} recordings')

    print(f'  {"":<46} {len(contents)} animals, '
          f'{sum(len(r) for _, r in contents)} recordings')

    if not check_timing:
        return

    # Named before a run that may take hours, as well as at the end of it
    untimed = [recording for _, recordings in contents for recording in recordings
               if lacks_frame_timing(recording)]
    if len(untimed) > 0:
        print(f'\n  {len(untimed)} recordings have imaging output but no frame timing in '
              f'their metadata.yaml, and are left out until they do:')
        for recording in untimed:
            print(f'    {recording}')


def _open(args) -> VxEntarchy:
    if os.path.exists(os.path.join(args.entarchy, CONFIG_NAME)):
        return VxEntarchy(args.entarchy)

    if not args.create:
        raise SystemExit(
            f'No entarchy at {args.entarchy}. Make one with "create", or pass '
            f'--create to make it here.')

    return VxEntarchy.open_or_create(args.entarchy, *_backend_config(args))


def _create(args) -> int:
    if os.path.exists(os.path.join(args.entarchy, CONFIG_NAME)):
        raise SystemExit(f'There is already an entarchy at {args.entarchy}.')

    backend, config = _backend_config(args)
    ent = VxEntarchy.create(args.entarchy, backend, config)

    print(f'\nCreated {args.entarchy} ({backend})')
    ent.backend.close()

    return 0


def _add(args) -> int:
    # Read the folder and the options before opening anything, so that a
    #  mistake in either is an error now rather than halfway through a run
    imaging = _imaging_from_args(args)
    contents = _contents(args.folder, args.limit)
    _print_contents(args.folder, contents, check_timing=imaging is not None)

    if args.dry_run:
        return 0

    ent = _open(args)

    try:
        experiment = ent.add_experiment(
            args.folder, imaging=imaging, name=args.name,
            with_video=not args.no_video, skip_broken=not args.stop_on_error,
            limit=args.limit)

        # Check against the folder rather than against a counter, so that a
        #  recording which failed for any reason is named here
        wanted = {(os.path.basename(animal_path), os.path.basename(recording))
                  for animal_path, recordings in contents for recording in recordings}
        present = {(recording.animal.id, recording.id)
                   for recording in experiment.recordings}
        missing = sorted(wanted - present)

        print(f'\n{experiment.id}: {len(experiment.animals)} animals, '
              f'{len(experiment.recordings)} recordings, '
              f'{len(experiment.rois)} ROIs')

        if missing:
            print(f'\n{len(missing)} of the folder did not make it in:')
            for animal_id, recording_id in missing:
                print(f'  {animal_id}/{recording_id}')

            return 1
    finally:
        ent.backend.close()

    return 0


def _scan(args) -> int:
    _print_contents(args.folder, _contents(args.folder, args.limit))

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m vxentarchy',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    create_parser = subparsers.add_parser(
        'create', help='make an empty entarchy')
    create_parser.add_argument('entarchy', help='directory to create')
    _add_backend_options(create_parser)

    add_parser = subparsers.add_parser(
        'add', help='ingest an experiment folder',
        description='Ingest <experiment>/<animal>/<recording>. Run it again '
                    'whenever the folder has grown: what is already in the '
                    'entarchy is skipped.')
    add_parser.add_argument('entarchy', help='entarchy directory')
    add_parser.add_argument('folder', help='experiment folder')
    add_parser.add_argument('--name', default=None,
                            help='experiment id (default: the folder name)')
    add_parser.add_argument('--create', action='store_true',
                            help='make the entarchy if it is not there yet')
    add_parser.add_argument('--dry-run', action='store_true',
                            help='say what would be ingested and stop')
    add_parser.add_argument('--limit', type=int, default=None,
                            help='at most this many recordings per animal, for a '
                                 'trial run before committing hours to it')
    add_parser.add_argument('--no-video', action='store_true',
                            help='leave the behaviour videos out')
    add_parser.add_argument('--stop-on-error', action='store_true',
                            help='end the run on the first folder that fails, '
                                 'instead of reporting it and carrying on')

    imaging_group = add_parser.add_argument_group(
        'imaging', 'which signals to take; how their frames were timed is read '
                   'from each recording folder\'s metadata.yaml')
    imaging_group.add_argument('--imaging', default='auto',
                               help='"auto" for every source found in the folder, '
                                    '"none" for stimulus and behaviour only, or a '
                                    'source name such as suite2p')

    _add_backend_options(add_parser)

    scan_parser = subparsers.add_parser(
        'scan', help='read an experiment folder without ingesting it')
    scan_parser.add_argument('folder', help='experiment folder')
    scan_parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == 'create':
        return _create(args)

    if args.command == 'add':
        return _add(args)

    return _scan(args)


if __name__ == '__main__':
    sys.exit(main())
