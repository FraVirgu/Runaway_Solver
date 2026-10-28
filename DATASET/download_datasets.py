#!/usr/bin/env python3
#
# Download the simulation datasets attached to the GitHub release
# "PlasmaSimulationRE" and verify them against their sha256 digests.
#
# Usage:
#   $ python download_datasets.py                     # all files, into DATASET/
#   $ python download_datasets.py output_CQ_M4.h5     # only selected files
#   $ python download_datasets.py --dir /some/path    # custom destination
#
# Files already present with the correct checksum are skipped.

import argparse
import hashlib
import os
import sys
import urllib.request

RELEASE_URL = 'https://github.com/FraVirgu/Runaway_Solver/releases/download/PlasmaSimulationRE'

# name -> sha256 (as shown on the release page)
DATASETS = {
    'output_CQ_M4.h5':              '4a8feffdd020fd9eeaf997a8071076ee90ac2ef6529e17e3eabc25eda34786c3',
    'output_injection_start_M4.h5': 'd518dc69074e0c2dc310e7bc560a5b1fcb604912ca356d0899f456bd3ef7f745',
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def progress(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100.0, 100.0 * done / total)
        sys.stdout.write(f'\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)')
        sys.stdout.flush()


def download(name, sha256, out_dir):
    dest = os.path.join(out_dir, name)

    if os.path.exists(dest):
        if sha256sum(dest) == sha256:
            print(f'{name}: already present, checksum OK')
            return dest
        print(f'{name}: checksum mismatch, downloading again')

    url = f'{RELEASE_URL}/{name}'
    tmp = dest + '.part'
    print(f'{name}: downloading {url}')
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()

    if sha256sum(tmp) != sha256:
        os.remove(tmp)
        raise RuntimeError(f'{name}: checksum mismatch after download')

    os.replace(tmp, dest)
    print(f'{name}: saved to {dest}')
    return dest


def main():
    parser = argparse.ArgumentParser(description='Download simulation datasets from the GitHub release.')
    parser.add_argument('files', nargs='*', help=f'files to download (default: all of {", ".join(DATASETS)})')
    parser.add_argument('--dir', default=OUT_DIR, help=f'destination directory (default: {OUT_DIR})')
    args = parser.parse_args()

    unknown = [f for f in args.files if f not in DATASETS]
    if unknown:
        parser.error(f'unknown file(s): {", ".join(unknown)}')

    os.makedirs(args.dir, exist_ok=True)
    for name in args.files or DATASETS:
        download(name, DATASETS[name], args.dir)


if __name__ == '__main__':
    main()
