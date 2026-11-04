"""Gzip the regime-map page's CSV downloads in a gallery checkout and rewrite their references.
Usage: compress_regime_map.py <repo>. Idempotent. Content is unchanged (gzip -9 of the same bytes)."""
import sys, gzip, json, re, pathlib, shutil
repo=pathlib.Path(sys.argv[1]); page=repo/'gallery/experiments/regime-map'
before=sum(p.stat().st_size for p in page.rglob('*') if p.is_file())
csvs=sorted(p for p in page.rglob('*.csv') if p.is_file())
html=(page/'index.html').read_text()
jsons={p:p.read_text() for p in page.rglob('*.json')}
for p in csvs:
    rel=p.relative_to(page).as_posix(); gz=p.with_name(p.name+'.gz')
    with open(p,'rb') as f, gzip.GzipFile(filename='',mode='wb',fileobj=open(gz,'wb'),compresslevel=9,mtime=0) as g: shutil.copyfileobj(f,g)
    p.unlink()
    html=html.replace(f'"{rel}"',f'"{rel}.gz"')
    for j in jsons:
        jd=j.parent
        if jd==p.parent: jsons[j]=jsons[j].replace(f'"{p.name}"',f'"{p.name}.gz"')
        jsons[j]=jsons[j].replace(f'"{rel}"',f'"{rel}.gz"')
note='<p><em>CSV downloads on this page are gzip-compressed (.csv.gz, identical content) so the gallery stays within its size limit (4 November 2026).</em></p>'
if note not in html:
    m=re.search(r'</h1>',html); html=html[:m.end()]+note+html[m.end():]
(page/'index.html').write_text(html)
for j,s in jsons.items(): json.loads(s); j.write_text(s)
# every local href/src on the page must resolve
bad=[v for v in re.findall(r'(?:href|src)="([^"#:?]+)"',html) if not v.startswith('../') and not v.endswith('/') and not (page/v).is_file()]
if bad: raise SystemExit(f'Unresolved page references: {bad[:10]}')
after=sum(p.stat().st_size for p in page.rglob('*') if p.is_file())
print(f'regime-map: {len(csvs)} CSVs compressed, {before/1024**2:.1f} -> {after/1024**2:.1f} MiB')
