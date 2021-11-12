"""Git utilities."""

import subprocess


def clone(remote, repo):
    return subprocess.run(['git', 'clone', '--recurse-submodules', remote, str(repo)], check=True)


def checkout(repo, commit):
    return subprocess.run(['git', 'checkout', str(commit)], cwd=str(repo), check=True)


def is_dirty(repo):
    p = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD'], check=False, cwd=str(repo))
    return p.returncode != 0


def branch(repo):
    p = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                       cwd=str(repo), stdout=subprocess.PIPE, encoding='ascii', check=True)
    return p.stdout.strip()
