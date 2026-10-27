"""Verify the final ZIP, payload manifest, and consolidated report structure."""
import hashlib
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'local-ner-bakeoff-submission.zip'
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        manifest = json.loads(archive.read('SUBMISSION_MANIFEST.json').decode('utf-8-sig'))
        listed = {entry['path'] for entry in manifest['files']}
        assert listed == names - {'SUBMISSION_MANIFEST.json'}
        for entry in manifest['files']:
            payload = archive.read(entry['path'])
            assert hashlib.sha256(payload).hexdigest() == entry['sha256'], entry['path']
            assert len(payload) == entry['bytes'], entry['path']
        assert not any(n.startswith(('.venv/', '.cache/', '.git/')) or '__pycache__' in n for n in names)
        report_bytes = archive.read('FULL_REPORT.html')
        assert report_bytes == (ROOT/'FULL_REPORT.html').read_bytes()
        report = report_bytes.decode('utf-8')
    HTMLParser().feed(report)
    ids = re.findall(r' id="([^"]+)"', report)
    links = re.findall(r'href="#([^"]+)"', report)
    assert len(ids) == len(set(ids))
    assert set(links).issubset(ids)
    questions = re.findall(r'<h3[^>]*>Q(\d+)\.', report)
    assert questions == list(map(str, range(1, 34)))
    assert len(links) == 28
    for expected in ('208 / 231', '176 / 198', '0.6380', '0.8889', 'post-review', 'interrupted after 13'):
        assert expected in report, expected
    print('Report words:', len((ROOT/'FULL_REPORT.md').read_text(encoding='utf-8').split()))
    print('Study questions:', len(questions))
    print('Navigation sections:', len(links))
    print(f'ZIP: {len(names)} files; {path.stat().st_size:,} bytes')
    print('All payload checksums, report navigation, and ZIP integrity verified.')


if __name__ == '__main__':
    main()
