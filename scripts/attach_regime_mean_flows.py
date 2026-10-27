"""Attach final, derived mean-flow comparisons to the existing regime gallery."""
import argparse
import datetime
import html
import json
from pathlib import Path
import shutil

START = '<!-- BEGIN MATCHED MEAN FLOWS -->'
END = '<!-- END MATCHED MEAN FLOWS -->'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    a = parser.parse_args()
    base = a.gallery / 'experiments/regime-map'
    target = base / 'mean-flows'
    target.mkdir(exist_ok=True)
    manifest = json.loads((a.figures / 'index-manifest.json').read_text())
    manifest['export_provenance'] = 'export-provenance.json'
    manifest['published_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for p in a.figures.iterdir():
        if p.suffix in ('.png', '.pdf', '.csv') and p.name != 'mean-profiles.csv':
            shutil.copyfile(p, target / p.name)
    shutil.copyfile(a.provenance, target / 'export-provenance.json')
    (target / 'index-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    groups = manifest['groups']
    anchor = next(g for g in groups if g['name'] == 'anchor' and g['seed'] == 1)
    def label(g):
        c = g['config']
        reynolds = c['eddy_rms'] * c['injection_wavelength'] / c['nu']
        name = {'primary':'Main case', 'seed':'Second seed'}.get(g['experiment'], g['experiment'].replace('_', ' ').capitalize())
        return f"{g['name']} · {1000*c['eddy_rms']:g} mm/s · α={c['alpha']:g} · Re={reynolds:g} · seed {g['seed']}"
    options = ''.join(f'<option value="{html.escape(g["slug"])}"'+(' selected' if g is anchor else '')+f'>{html.escape(label(g))}</option>' for g in groups)
    downloads = ''.join(f'<tr><td>{html.escape(label(g))}</td><td><a href="mean-flows/{g["profile_plot"]["pdf"]}">Profiles PDF</a> · <a href="mean-flows/{g["profiles_csv"]}">All times CSV</a></td><td><a href="mean-flows/{g["field_plot"]["pdf"]}">Flow maps PDF</a></td></tr>' for g in groups)
    data = {g['slug']: {'label':label(g),'profiles':g['profile_plot'],'fields':g['field_plot']} for g in groups}
    block = f'''{START}
<section class="section" id="mean-flows" aria-labelledby="mean-flows-title">
<p class="eyebrow">Direct comparison of the saved mean flows</p><h2 id="mean-flows-title">2D, weakly 3D and finite-3D mean flow</h2>
<p>The matched runs share the same initial mean flow and in-plane eddies. Black shows 2D; dashed turquoise shows a weak transverse disturbance (initial transverse/2D eddy energy η=10⁻⁶); orange shows a finite disturbance (η=1). These labels describe the initial perturbations. Finite disturbances add energy.</p>
<p>All currents below are <strong>Eulerian</strong>. The profiles average over both horizontal directions; the flow maps average only over the transverse direction y. The imposed initial current and vorticity are retained. Values are shown at the saved cell centres, without spatial filtering.</p>
<label for="mean-flow-case"><strong>Choose a matched comparison</strong></label><br>
<select id="mean-flow-case" style="max-width:100%;font:inherit;padding:10px;margin:8px 0 24px;border:1px solid #acc0c3;border-radius:7px;background:white;color:#142d3b">{options}</select>
<figure id="mean-profile-comparison"><img id="mean-profile-image" src="mean-flows/{anchor['profile_plot']['png']}" alt="Central case: overlaid 2D, weakly 3D and finite-3D horizontal mean-current profiles">
<figcaption><strong>Depth profiles.</strong> Left: total horizontal mean current, including the initial shear (grey dotted). Right: change from the initial horizontal mean. The lower panels enlarge the upper two Stokes depths. Overlapping black and turquoise curves mean the 2D and weakly 3D profiles agree at this scale. <a id="mean-profile-pdf" href="mean-flows/{anchor['profile_plot']['pdf']}">Download profiles PDF</a>.</figcaption></figure>
<figure id="mean-field-comparison"><img id="mean-field-image" loading="lazy" src="mean-flows/{anchor['field_plot']['png']}" alt="Central case: side-by-side transverse-averaged x-z Eulerian flow maps">
<figcaption><strong>Flow maps at 600 s.</strong> Columns show 2D, weakly 3D and finite 3D. Top: total transverse mean U(x,z). Bottom: U(x,z) minus the <em>initial horizontal mean profile</em>, exposing the remaining in-plane eddies and changed jets. This subtraction retains the evolving eddies; it does not subtract the initial x-z eddy field. Colours share one scale across each row. <a id="mean-field-pdf" href="mean-flows/{anchor['field_plot']['pdf']}">Download flow maps PDF</a>.</figcaption></figure>
<details><summary>Compare the mean-current changes across all seven primary cases</summary>
<figure style="margin-top:20px"><img loading="lazy" src="mean-flows/{manifest['overview']['png']}" alt="Final horizontal mean-current changes for all seven primary cases"><figcaption>Each panel compares the three matched runs at 600 s. Horizontal ranges vary across panels so the current changes remain visible. <a href="mean-flows/{manifest['overview']['pdf']}">Download seven-case PDF</a>.</figcaption></figure></details>
<details><summary>Follow the central profiles through time</summary><figure style="margin-top:20px"><img loading="lazy" src="mean-flows/{manifest['anchor_evolution']['png']}" alt="Central Eulerian mean-current profiles at 0,120,300 and 600 seconds"><figcaption>Total current and its change from the initial profile at the four indicated times. <a href="mean-flows/{manifest['anchor_evolution']['pdf']}">Download evolution PDF</a>.</figcaption></figure></details>
<details><summary>All 14 matched comparisons: main cases, second seeds and controls</summary><div class="table-wrap" style="margin-top:16px"><table><thead><tr><th>Comparison</th><th>Depth profiles</th><th>x-z mean fields</th></tr></thead><tbody>{downloads}</tbody></table></div></details>
<div class="downloads"><a href="mean-flows/{manifest['profiles_csv']}">Final horizontal mean profiles (CSV, SI units)</a><a href="mean-flows/index-manifest.json">Plot definitions and provenance</a></div>
</section>
<script>(function(){{const comparisons={json.dumps(data)};document.getElementById('mean-flow-case').addEventListener('change',function(){{const g=comparisons[this.value];const p=document.getElementById('mean-profile-image');const f=document.getElementById('mean-field-image');p.src='mean-flows/'+g.profiles.png;p.alt=g.label+': matched horizontal mean profiles';f.src='mean-flows/'+g.fields.png;f.alt=g.label+': matched transverse mean fields';document.getElementById('mean-profile-pdf').href='mean-flows/'+g.profiles.pdf;document.getElementById('mean-field-pdf').href='mean-flows/'+g.fields.pdf;}});}})();</script>
{END}'''
    page = base / 'index.html'
    text = page.read_text()
    if START in text:
        start, tail = text.split(START, 1)
        _, end = tail.split(END, 1)
        text = start + block + end
    else:
        anchor_markup = '<section class="section" aria-labelledby="results-title">'
        if anchor_markup not in text:
            raise ValueError('Regime results section was not found')
        text = text.replace(anchor_markup, block + '\n' + anchor_markup, 1)
    link = '<a href="#mean-flows">Mean-flow comparisons</a>'
    if link not in text:
        text = text.replace('<div class="tabs">', '<div class="tabs">' + link, 1)
    page.write_text(text)
    print('Attached', len(groups), 'verified mean-flow triplets.')

if __name__ == '__main__':
    main()
