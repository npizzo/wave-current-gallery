# Surface waves & turbulence

[Open the research gallery](https://npizzo.github.io/wave-current-gallery/)

A results gallery for the bounded wave-packet laboratory reproduction based on
[NumericalWaveTanks.jl](https://github.com/glwagner/NumericalWaveTanks.jl/tree/9ff503ca9bf4672a30f32c5dcfcc336f0be6faed).
It shows completed wave/no-wave realizations and, after the campaign analysis
finishes, the seven-seed laboratory comparison and momentum-flux diagnostics.
The initial page explicitly awaits its first Unity synchronization.

## How updates work

Unity runs small CPU publishing jobs after each array member finishes and after
the final analysis. Each job verifies completion records and source hashes,
renders completed results, and pushes only the derived gallery assets. GitHub
Actions then deploys `gallery/` to GitHub Pages. An initial publishing job catches
results that were already finished when the updater was attached.

The publisher lives in a separate checkout. It does not change the scientific
campaign, submit simulations, or read unfinished native output. Large JLD2 files
remain on Unity. A repository-specific deploy key stays on Unity; no Unity login
key is stored on GitHub. Updates wait for CPU queue availability and deployment,
so the page is a dated snapshot rather than a live scheduler display.

## One-time Unity setup

Clone this repository beside, rather than inside, the scientific campaign:

```bash
git clone https://github.com/npizzo/wave-current-gallery.git
cd wave-current-gallery
bash unity/setup-key.sh
```

Add the printed **public** key in this repository's **Settings → Deploy keys →
Add deploy key**, titled `Unity gallery publisher`, with **Allow write access**
checked. The private key remains in the ignored `.unity-publisher/` directory.
Then attach the publisher to the existing campaign, supplying its absolute path:

```bash
python3 unity/enable-updates.py --campaign /absolute/path/to/unity-packet-reproduction
```

This reads the existing array and analysis IDs from `data/submission.json`, checks
GitHub write access and Slurm state, and records the publishing job IDs in
`.unity-publisher/updates.json`. Do not run it twice. It is safe to log out.
The repository's Pages source must be **GitHub Actions**.

## Check or retry a publication

```bash
cat .unity-publisher/updates.json
squeue -u "$USER"
```

Publisher logs are in `.unity-publisher/logs/publish-JOBID.out` and `.err`.
If a publishing job fails, the simulations and previous website remain intact.
After resolving the cause, submit one publication snapshot from this checkout:

```bash
sbatch --output="$PWD/.unity-publisher/logs/publish-%j.out" \
  --error="$PWD/.unity-publisher/logs/publish-%j.err" \
  unity/publish.sbatch "$PWD" /absolute/path/to/unity-packet-reproduction \
  npizzo/wave-current-gallery
```

Do not resubmit the scientific array or analysis to refresh the website. To stop
future gallery jobs, cancel only the publisher IDs in `updates.json`. Revoke the
repository deploy key when publication is no longer needed.

## Interpretation and provenance

The seven retained seeds are 1, 2, 3, 4, 6, 7, 8, reproducing the upstream
selection after instability in seeds 5 and 9. Individual profiles are not the
final ensemble comparison. Eulerian currents, full Lagrangian RMS velocities,
ensemble-incoherent dimensionality and momentum flux are labelled separately.
Completion does not establish convergence, Taylor–Proudman balance, or tracer
mixing suppression. See [diagnostic definitions](docs/diagnostics.txt).

## Local checks

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
python3 scripts/check_gallery.py gallery
python3 -m http.server 8767 --directory gallery
```

The tests use temporary synthetic fixtures; these are never published. Production
extraction uses the existing campaign's Julia and Python environments. GitHub
Pages receives only the checked static `gallery/` directory.
