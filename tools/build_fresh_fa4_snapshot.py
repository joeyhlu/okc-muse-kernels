"""Build an ignored, namespace-relocated LAB snapshot from a pinned local git object.

No network, installation, loaded-module editing, or GPU activity. This large
research snapshot is not a competition submission; a winner requires vendoring.
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SHA = '303f735573162381de2dc22dc202be6aa594c264'
DEST = ROOT/'kernels/lab/fresh_fa4'


def main():
    blob = subprocess.check_output(['git','-C',str(ROOT/'repos/flash-attention'),
                                   'archive',SHA,'flash_attn/cute'])
    records=[]
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'__init__.py').write_text('# Pinned research snapshot; see manifest.json.\n')
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive:
            parts=Path(member.name).parts
            if not member.isfile() or not member.name.endswith('.py'):
                continue
            if parts[:2] != ('flash_attn','cute') or '..' in parts:
                raise ValueError(member.name)
            raw=archive.extractfile(member).read()
            # Imports and qualified references stay inside the lab namespace.
            data=raw.replace(b'flash_attn.cute',b'fresh_fa4.cute')
            target=DEST.joinpath(*parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            records.append(dict(path=str(target.relative_to(DEST)),
                original_sha256=hashlib.sha256(raw).hexdigest(),lab_sha256=hashlib.sha256(data).hexdigest()))
    (DEST/'manifest.json').write_text(json.dumps(dict(commit=SHA,namespace='fresh_fa4',files=records),indent=2))
    print('SNAPSHOT',SHA,len(records),'Python files')


if __name__=='__main__':main()
