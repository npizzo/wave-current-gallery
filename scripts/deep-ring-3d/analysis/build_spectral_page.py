"""Build gallery-page/spectral-exclusion/ (2D spectral-exclusion study) from figs/, and link it from gallery-page/index.html."""
import os, shutil, json
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); F=f'{ROOT}/figs'; P=f'{ROOT}/gallery-page/spectral-exclusion'
shutil.rmtree(P,ignore_errors=True); os.makedirs(P+'/data')
def webp(src,name,w=1800):
    with Image.open(f'{F}/{src}') as im:
        im=im.convert('RGB'); im.thumbnail((w,w)); im.save(f'{P}/{name}.webp',quality=82,method=6)
    return f'{name}.webp'
IMG={k:webp(v,k) for k,v in dict(v_deep='ens-vallis/vallis2-deep.png',v_ubeta='ens-vallis/vallis2-ubeta.png',v_none='ens-vallis/vallis2-none.png',
     grid='ens-vallis/ens-grid.png',u35='ens-vallis/u-t35.png',jf='exp2d/jet-fraction.png',fields='exp2d/fields-final.png',
     collapse='exp2d/kx-collapse.png',composite='composite-exp/composite.png',fullnone='ens-vallis/fullbox-s2d-none-k27-lx4.png',
     latespec='long2d/spectra.png',jf9='long2d/jet-fraction.png').items()}
for src,dst in [('long2d/summary.json','long-runs.json'),('exp2d/summary.json','profiles-kx-collapse.json'),('composite-exp/composite.json','composite.json'),('ens-vallis/summary.json','ensemble.json')]:
    shutil.copy(f'{F}/{src}',f'{P}/data/{dst}')
def fig(k,cap,alt): return f'<figure><img src="{IMG[k]}" alt="{alt}"><figcaption>{cap}</figcaption></figure>'
css='body{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#173047;background:#f6f8fa}img{width:100%;background:white;margin:12px 0}figure{margin:28px 0}figcaption{font-size:15px;color:#3b5368}section{margin:50px 0}a{color:#135da8}table{border-collapse:collapse;font-size:15px;background:white}td,th{border-bottom:1px solid #d8dee4;padding:4px 10px;text-align:right}td:first-child,th:first-child{text-align:left}'
html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Spectral exclusion in 2D CL</title><style>{css}</style>
<a href="../">← Deep-water jets in 3D CL</a> · <a href="../../../">Wave–current gallery</a>
<h1>Is there a Rhines-type excluded region in the CL spectra?</h1>
<p>A 2D study of where the inverse cascade can and cannot put energy when the Craik–Leibovich (CL) vortex force supplies an effective β, β<sub>eff</sub>=∂<sub>zz</sub>u<sup>s</sup>, compared with the classic β-plane picture (Vallis 2017, Figs 12.3–12.4). All runs are 2D (x–z) CL in Oceananigans: periodic in x with L<sub>x</sub>=4, free-slip walls at z=−1 and the surface z=0, 2048×512 grid, WENO-9 implicit LES, prognostic Lagrangian-mean velocity. The initial condition is a narrow Vallis ring at 27 cycles with r<sub>b</sub>=1/9, which keeps the same u<sub>0</sub> and Rhines scale (≈9 cycles) as the 9-cycle runs on the <a href="../">parent page</a> but starts the eddies three times smaller, leaving room for an inverse cascade. T<sub>e</sub> below is the eddy time of this 27-cycle ring.</p>
<table><tr><th>profile</th><th>β<sub>eff</sub>(z)</th></tr>
<tr><td>deep water (Wagner–Craig “deep”)</td><td>2e<sup>8z</sup></td></tr><tr><td>uniform β</td><td>½</td></tr>
<tr><td>slower decay</td><td>∝e<sup>2z</sup>, e<sup>4z</sup> (equal to ½ at z=−0.17)</td></tr><tr><td>no waves</td><td>0</td></tr></table>
<p>Long runs (seed 1) go to 3000 T<sub>e</sub>. An ensemble of 8 random realizations of the initial ring (“seeds”: same ring spectrum, different random phases) was run for deep water, uniform β and no waves to 100 T<sub>e</sub> with output every T<sub>e</sub>. Spectra are of the y-mean (u,w), reflected evenly/oddly about the walls; deep-water spectra use the jet layer z&gt;−0.5 (bottom quarter tapered, zero-padded in z for display).</p>
<h2>Findings</h2><ul>
<li><b>The excluded region is there, as a transient during the inverse cascade.</b> At 15–20 T<sub>e</sub> the ensemble-mean spectra of both wave cases show a dumbbell-shaped hole bounded by κ=κ<sub>R</sub>|cosθ|<sup>3/5</sup>, with energy reaching small k only along the k<sub>z</sub> axis. The no-wave control fills in isotropically.</li>
<li><b>Late times:</b> energy is confined to a narrow band around k<sub>x</sub>=0. Uniform β keeps two clean empty lobes; in deep water the lobes are partly filled.</li>
<li><b>Jets in u come much later than the spectral signature.</b> At 35 T<sub>e</sub> none of the three cases has visible bands (1.3–1.5% of u-energy at k<sub>x</sub>=0 in all three, 8-seed mean); bands are clear after several hundred T<sub>e</sub>.</li>
<li><b>Why deep water is less clean:</b> β<sub>eff</sub>=2e<sup>8z</sup> changes by ≈60% over one Rhines wavelength, so a single spectrum mixes dumbbells of different sizes, and below z≈−0.5 (β≈0) the flow is ordinary 2D turbulence (vortices and a bottom current). Scaling k<sub>x</sub> at each depth by the local Rhines wavenumber collapses the spectra when β varies slowly: the energy fraction at k<sub>x</sub>≤0.5k<sub>β</sub>(z) is 0.997 (uniform β), 0.99 (e<sup>2z</sup>), 0.98 (e<sup>4z</sup>), 0.97 (deep, spread 0.89–0.99 with depth) and 0.81 (no waves, no collapse).</li>
<li><b>Measure jets carefully:</b> the strict x-mean misses slowly tilted bands: final jet fraction is 0.68 in deep water (surface jets plus a bottom current) but 0.06 for uniform β, whose bands carry 47% of the energy at the box scale k<sub>x</sub>=0.25.</li></ul>
<section><h2>Vallis-style figures</h2>
<p>Top: energy in (k<sub>x</sub>,k<sub>z</sub>), shades at 0.05, 0.15, 0.35, 0.65 of the maximum; the right panel is zoomed and on a log grey scale (10<sup>−3</sup>–1). Left and middle: mean of 8 seeds; right: seed 1, averaged over 800–1000 T<sub>e</sub>. Red dashed: κ=κ<sub>R</sub>|cosθ|<sup>3/5</sup>, κ<sub>R</sub>=(|β|/𝒯<sup>1/3</sup>)<sup>3/5</sup> (constant-flux estimate), with the transfer rate estimated as 𝒯=ũ<sup>3</sup>κ<sub>e</sub> (ũ=(2KE)<sup>1/2</sup>, κ<sub>e</sub> the energy-weighted mean |k| of the panel); κ<sub>R</sub>≈7.2–7.5 cycles early and ≈10 cycles late. Bottom: u/u<sub>0</sub> for seed 1, 0&lt;x&lt;2.</p>
{fig('v_deep','Deep water, β<sub>eff</sub>=2e<sup>8z</sup> (spectra over z&gt;−0.5, marked by the dotted line).','deep water')}
{fig('v_ubeta','Uniform β=½: the textbook case.','uniform beta')}
{fig('v_none','No waves (control). The late u panel looks banded only because u=−∂<sub>z</sub>ψ of a depth-filling cell changes sign across its centre; the full box (below) is a patchwork of large cells and coherent vortices, with 2.4% of the energy at k<sub>x</sub>=0.','no waves')}
{fig('fullnone','No waves, u/u<sub>0</sub> at 900 T<sub>e</sub> over the whole box: decaying 2D turbulence, not zonal flow.','no waves full box')}</section>
<section><h2>Time sequence of the spectra</h2>{fig('grid','8-seed mean spectra at 0–60 T<sub>e</sub> (rows: uniform β, deep water z&gt;−0.5, no waves); shades at 10<sup>−2</sup>–10<sup>−0.5</sup> of each panel’s maximum; red dashed: fixed-velocity Rhines curve k<sub>β</sub>|cosθ|<sup>1/2</sup> used in this earlier version.','spectra time sequence')}</section>
<section><h2>When do jets appear?</h2>{fig('u35','u/u<sub>0</sub> at 35 T<sub>e</sub> (seed 1, full box) with ⟨u⟩<sub>x</sub>: no bands yet in any case.','u at 35 Te')}
{fig('jf','Jet fraction ½⟨U<sub>L</sub><sup>2</sup>⟩/KE (strict x-mean) vs time, all 27-cycle profiles, full depth and z&gt;−0.5; time in T<sub>e</sub> of the 9-cycle ring (×3 for the 27-cycle T<sub>e</sub>).','jet fraction')}
{fig('fields','u/u<sub>0</sub> at the end of each long run (3000 T<sub>e</sub>).','final fields')}
{fig('jf9','Jet fraction vs time including the 9-cycle ring at L<sub>x</sub>=1 and 4 (these agree once run long enough).','jet fraction 9-cycle')}</section>
<section><h2>Depth-varying β and the local Rhines scale</h2>{fig('collapse','k<sub>x</sub> spectra of the y-mean field at nine depths, scaled by the local k<sub>β</sub>(z)=(β(z)/ũ(z))<sup>1/2</sup>/2π (last 20% of each run); bottom: cumulative energy. Dotted: k<sub>x</sub>=0.5k<sub>β</sub>.','kx collapse')}
{fig('composite','Annulus-normalized spectra log<sub>2</sub>[E/⟨E⟩<sub>|k|</sub>], last 20% of each run. Top: one transform over z&gt;−0.5 scaled by the layer-mean k<sub>β</sub>; bottom: composite of depth windows each scaled by its own k<sub>β</sub>(z). The inside/outside numbers printed on the panels are not a reliable dumbbell test (see the parent discussion); read the pictures.','composite spectra')}
{fig('latespec','Late-time (last 20%) spectra, log and annulus-normalized, for the 9- and 27-cycle rings.','late spectra')}</section>
<section><h2>Caveats</h2><ul><li>2D and decaying: the excluded region is a transient of the inverse cascade, and κ<sub>R</sub> uses a dimensional estimate of 𝒯 rather than a measured flux.</li>
<li>Late-time panels are a single realization; the ensemble stops at 100 T<sub>e</sub>.</li><li>The deep-water bottom layer (β≈0) develops a 2D condensate that does not occur in 3D (see the 3D results on the <a href="../">parent page</a>).</li></ul></section>
<p>Data: <a href="data/ensemble.json">ensemble</a> · <a href="data/long-runs.json">long runs</a> · <a href="data/profiles-kx-collapse.json">k<sub>x</sub> collapse</a> · <a href="data/composite.json">composite</a> · source in the repository under <code>scripts/deep-ring-3d</code></p>'''
open(f'{P}/index.html','w').write(html)
idx=f'{ROOT}/gallery-page/index.html'; s=open(idx).read(); link='<p><b>New:</b> <a href="spectral-exclusion/">Spectral exclusion in 2D CL — Vallis-style spectra, deep water vs uniform β</a></p>'
if 'spectral-exclusion/' not in s:
    i=s.index('</h1>')+5; s=s[:i]+link+s[i:]; open(idx,'w').write(s)
tot=sum(os.path.getsize(os.path.join(d,f)) for d,_,fs in os.walk(P) for f in fs); print('subpage MB',round(tot/1e6,2))
