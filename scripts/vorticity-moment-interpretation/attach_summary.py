"""Append the completed-campaign interpretation without regenerating run results."""
from pathlib import Path
import argparse
import re
import shutil

BEGIN = '<!-- BEGIN VORTICITY CAMPAIGN CONCLUSIONS -->'
END = '<!-- END VORTICITY CAMPAIGN CONCLUSIONS -->'
CSS = '''
/* Completed-campaign interpretation */
#conclusions{scroll-margin-top:24px;padding-bottom:44px;border-top:2px solid #acc0c3;margin-top:26px}
#conclusions .conclusion-lead{background:#eaf1ee;border-left:4px solid #12687b;padding:22px 25px;margin-bottom:30px;border-radius:0 10px 10px 0}
#conclusions>h3{font-size:22px;margin-top:34px;margin-bottom:14px}
#conclusions>p,#conclusions .conclusion-lead p{max-width:1030px;font-size:15px}
#conclusions .metric-definitions{display:grid;grid-template-columns:1fr 1fr;gap:15px 28px;background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:24px}
#conclusions .metric-definitions p{font-size:14px;margin:0}
#conclusions caption{text-align:left;font-size:13px;font-weight:600;color:var(--ink);padding:13px 14px;background:#edf2ef}
#conclusions .table-wrap{margin-bottom:20px}
#conclusions td:first-child{white-space:normal;min-width:175px}
#conclusions figure{margin-top:22px}
#conclusions .small{font-size:12px}
#conclusions .interpretation{font-size:15px;margin-top:24px}
.conclusion-jump{display:inline-block;font-size:13px;margin-top:14px;font-weight:600}
@media(max-width:750px){#conclusions .metric-definitions{grid-template-columns:1fr}#conclusions .conclusion-lead{padding:18px}#conclusions>h3{font-size:20px}}
'''

def attach(gallery, artifacts):
    target = gallery/'experiments/vorticity-moments'
    page = target/'index.html'
    original = page.read_text()
    section = (artifacts/'conclusions-section.html').read_text()
    if BEGIN in original:
        if original.count(BEGIN)!=1 or original.count(END)!=1:
            raise ValueError('Ambiguous existing conclusion section')
        updated = re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),lambda _:section.strip(),original,flags=re.S)
    else:
        if original.count('</main>')!=1:raise ValueError('Expected one main element')
        updated = original.replace('</main>',section+'\n</main>',1)
    if 'class="conclusion-jump"' not in updated:
        needle='<section class="questions">'
        if needle not in updated:raise ValueError('Expected existing questions section')
        updated=updated.replace(needle,'<a class="conclusion-jump" href="#conclusions">Read the completed-campaign conclusions and parameter summary ↓</a>\n'+needle,1)
    page.write_text(updated)
    style=target/'style.css'
    css=style.read_text();marker='/* Completed-campaign interpretation */'
    if marker in css:css=css[:css.index(marker)].rstrip()+'\n'
    style.write_text(css+CSS)
    source=artifacts/'site/interpretation'
    dest=target/'interpretation';dest.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,dest/path.name)
    script_dest=gallery.parent/'scripts/vorticity-moment-interpretation'
    script_dest.mkdir(parents=True,exist_ok=True)
    for name in ('build_summary_figures.py','attach_summary.py','conclusions-section.html'):
        shutil.copy2(artifacts/name,script_dest/name)
    return page

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery',required=True,type=Path)
    parser.add_argument('--artifacts',required=True,type=Path)
    args=parser.parse_args();print(attach(args.gallery,args.artifacts))
