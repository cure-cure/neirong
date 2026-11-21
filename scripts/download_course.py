"""Download only the user-provided public course share; preserve source hashes."""
from pathlib import Path
import urllib.request, urllib.parse, json, hashlib, concurrent.futures, os

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get('LAB_COURSE_BASE_URL', '')
SHARE = os.environ.get('LAB_COURSE_SHARE_ID', '')

def api(route, **params):
    url = BASE + 'explorer/share/' + route + '&' + urllib.parse.urlencode(dict(shareID=SHARE, **params))
    return url

def listing(path):
    obj = json.load(urllib.request.urlopen(api('pathList', path=path), timeout=45))
    if not obj.get('code'): raise RuntimeError(obj)
    data = obj['data']; files = data['fileList']
    for folder in data['folderList']: files += listing(folder['path'])
    return files

def download(item):
    relative = item['path'].split('}/', 1)[1]
    dest = ROOT / 'data/raw/course' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = api('fileOut', path=item['path'])
    if not dest.exists() or dest.stat().st_size != int(item['size']):
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        if len(content) != int(item['size']): raise RuntimeError(f'Size mismatch: {relative}, {content[:100]!r}')
        dest.write_bytes(content)
    return {'path': dest.relative_to(ROOT).as_posix(), 'bytes': dest.stat().st_size,
            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'source': url}

if __name__ == '__main__':
    if not BASE or not SHARE:raise SystemExit('Set LAB_COURSE_BASE_URL and LAB_COURSE_SHARE_ID from your authorized data source.')
    files = listing('{shareItemLink:' + SHARE + '}/')
    print('Discovered', len(files), 'files', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = []
        for row in pool.map(download, files):
            results.append(row)
            if len(results) % 25 == 0: print('Downloaded', len(results), flush=True)
    (ROOT/'data/course_sources.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Complete:', len(results), 'files', sum(x['bytes'] for x in results), 'bytes', flush=True)
