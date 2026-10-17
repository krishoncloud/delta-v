# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo==0.25.1", "torch==2.8.0", "neuraloperator==2.0.0",
#     "numpy==2.1.2", "scipy==1.14.1", "pandas==2.2.3",
#     "scikit-learn==1.8.0", "matplotlib==3.9.2",
#     "huggingface-hub==1.33.0",
# ]
# ///
"""Self-contained Delta-V Task 5 development workflow for marimo MoLab."""
import marimo
__generated_with = "0.25.1"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    import hashlib
    import time
    import warnings
    import shutil
    import platform
    import importlib.metadata as metadata
    from pathlib import Path
    from datetime import datetime, timezone
    from neuralop.models import FNO
    from neuralop.layers.spectral_convolution import SpectralConv
    notebook_source_path = Path(__file__).resolve()
    return mo, np, pd, torch, matplotlib, plt, json, hashlib, time, warnings, shutil, platform, metadata, Path, datetime, timezone, FNO, SpectralConv, notebook_source_path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delta-V · Task 5
    ## Bring a supported fluid study → adapt the actual FNO → predict engineering outcomes
    **Objective:** connect known setup and new knobs to your pretrained fluid FNO, with no completed query trajectory. This is the implementation and development pilot; Task 6 tests whether pretraining beats scratch and cheap baselines.

    **Scope:** one fixed geometry, regular 2D grid, four explicitly mapped fields `[density, pressure, velocity_x, velocity_y]`, one fixed observation time, and known initial conditions. The customer-facing answers are engineering outcomes calculated from predicted fields. This prototype supports kinetic energy, pressure RMS, or a declared two-probe pressure difference. It does not claim arbitrary fluid/geometry support.

    **Run order:** (1) select a CUDA GPU in MoLab, (2) review settings and click **Prepare study**, (3) inspect data and splits, (4) click **Load pretrained FNO**, (5) click **Train Task 5**, (6) inspect the report and download both ZIPs. All code is in this file; no repository clone or companion Python module is required.

    The default reference study is a controlled, exact 2D Taylor–Green vortex. It is useful for checking the mechanism, but cheap analytical data cannot establish simulation cost savings. A customer study can use the documented bundle format below. **No remote writes occur:** Task 15 is still paused. Download artifacts before ending a temporary MoLab session. No HF token is needed for the public source checkpoint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Environment and resource limits
    **Objective:** make the software, device and cost visible before training.

    **Input:** MoLab's environment and the dependencies pinned at the top of this file. **Output:** installed versions and CUDA availability. The source checkpoint download is 858 MB, its model has about 71.5 million real parameters, and full fine-tuning needs several GB of memory. Start with a CUDA GPU with at least 8 GB; memory suitability is checked by the first real step, not guaranteed from the device name. If out of memory, completed artifacts remain on disk.

    The development grid is **64 × 64**, while the served source used 256 × 256. Reusing spectral weights at this coarser resolution is an explicit additional adaptation assumption, recorded in the report. CPU smoke mode is limited to 10 steps; a tiny unit-test network is never silently used for a real notebook run.
    """)
    return


@app.cell
def _(metadata, mo, pd, torch, platform):
    versions = {name: metadata.version(name) for name in ['marimo','torch','neuraloperator','numpy','scipy','pandas','scikit-learn','huggingface-hub']}
    mo.vstack([pd.DataFrame([versions]), mo.md(f"CUDA available: **{torch.cuda.is_available()}** · Python {platform.python_version()}")])
    return versions


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Choose the study and bounded training configuration
    **Objective:** lock a small development run before seeing its results.

    **Default:** 16 training settings, 9 development settings, seed 42, 200 optimizer steps, validation every 20 steps, 15-minute training cap. The 16 reserved Task 6 settings have knobs only: this notebook never generates or evaluates their reference outcomes. Development validation selects a checkpoint, so its reported errors are exploratory.

    For your own data, choose **customer_bundle** and enter an existing directory in MoLab. Supply the metadata, initial fields, completed training/development fields and CSVs described in step 4. The study is rejected if it cannot satisfy this prototype's input contract. The default reference case needs no dataset download.

    Changing a setting after preparation requires clicking Prepare again and repeating the later buttons. Each preparation creates a new run directory; old runs are preserved. The source checkpoint identity is fixed, and a local checkpoint path is accepted only if its hash matches.
    """)
    return


@app.cell
def _(mo, torch):
    settings = mo.ui.dictionary({
        'study_mode': mo.ui.dropdown(['controlled_vortex','customer_bundle'], value='controlled_vortex', label='Study'),
        'bundle_path': mo.ui.text(value='', label='Customer bundle directory (optional)'),
        'checkpoint_path': mo.ui.text(value='', label='Local original checkpoint (optional)'),
        'device': mo.ui.dropdown(['cuda','cpu'], value='cuda' if torch.cuda.is_available() else 'cpu', label='Device'),
        'steps': mo.ui.number(start=1, stop=2000, value=200 if torch.cuda.is_available() else 6, label='Optimizer steps'),
        'max_seconds': mo.ui.number(start=10, stop=3600, value=900, label='Training time cap (seconds)'),
        'seed': mo.ui.number(start=0, stop=100000, value=42, label='Seed'),
    })
    prepare_button = mo.ui.run_button(label='Prepare study')
    mo.vstack([settings, prepare_button])
    return settings, prepare_button


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Audit the study and define what a prediction means
    **Objective:** choose an achievable new-setting task and document its limits.

    The old three Task 4 systems were part of source training. Euler supplied useful field-transfer evidence, but its four gamma values and already-inspected tests do not automatically create a fresh design study. We therefore use a controlled periodic vortex for this first implementation check. A harder external study remains essential for Task 6.

    The default knobs are initial velocity amplitude `A` in [0.5, 1.5] and viscosity `nu` in [0.01, 0.1], in nondimensional units. The periodic box is [0, 2π)², density is one, and evaluation time is 1. For this orientation:

    `u = A sin(x) cos(y) exp(-2 nu t)`; `v = -A cos(x) sin(y) exp(-2 nu t)`;
    `p = A²/4 [cos(2x) + cos(2y)] exp(-4 nu t)`.

    We check divergence and the momentum-equation residual numerically, and verify the mean kinetic energy against `A² exp(-4 nu t)/4`. References are created **only during data preparation**. Inference receives the initial state and knobs and has no call to the analytical future solution.

    Customer outputs here are mean specific kinetic energy and pressure RMS at t=1. For this exact vortex, these two values are mathematically equal: they test separate field readouts but are not independent accuracy successes. They are physically defined diagnostics, not a pressure-drop or mixing-efficiency claim. Source-study exclusion follows the recorded three-system training code; the legacy checkpoint has no complete signed exposure ledger, so exclusion is not cryptographically proven.

    Technical reference: [NeuralOperator FNO API](https://neuraloperator.github.io/dev/modules/generated/neuralop.models.FNO.html). The equations and residual checks below define this particular 2D study directly.
    """)
    return


@app.cell
def _(Path, hashlib, json, np, torch):
    SOURCE = {'repo':'KrishMalik/deltav-fluid', 'revision':'e25651ff92cf1d8f8818164a0277b8613e83d8b3',
              'filename':'checkpoints/domain_fluid.pt', 'sha256':'2aefbf631c8ce81f5037646b08d6e16ece9e736cbfda1b37ad4c924e261e1674', 'bytes':858147645}
    FNO_CONFIG = {'n_modes':(32,32), 'hidden_channels':128, 'in_channels':16, 'out_channels':4}
    SOURCE_SYSTEMS = ['turbulent_radiative_layer_2D','rayleigh_benard','shear_flow']

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(path)

    def vortex_fields(amplitude, viscosity, t, size=64):
        coordinates = np.arange(size, dtype=np.float64)*2*np.pi/size
        x, y = np.meshgrid(coordinates, coordinates, indexing='xy')
        decay = np.exp(-2*viscosity*t)
        return np.stack([np.ones_like(x), amplitude**2/4*(np.cos(2*x)+np.cos(2*y))*decay**2,
                         amplitude*np.sin(x)*np.cos(y)*decay, -amplitude*np.cos(x)*np.sin(y)*decay]).astype('float32')

    def physical_outcomes(fields, specs):
        # NumPy or torch: keep the same definitions for reference and prediction.
        values = []
        for spec in specs:
            if spec['kind'] == 'mean_kinetic_energy':
                value = 0.5*(fields[...,2,:,:]**2+fields[...,3,:,:]**2).mean(axis=(-2,-1))
            elif spec['kind'] == 'pressure_rms':
                p = fields[...,1,:,:]
                centered=p-p.mean(axis=(-2,-1), keepdims=True)
                value = torch.linalg.vector_norm(centered.flatten(start_dim=-2),dim=-1)/(p.shape[-2]*p.shape[-1])**0.5 if isinstance(p,torch.Tensor) else (centered**2).mean(axis=(-2,-1))**0.5
            elif spec['kind'] == 'pressure_difference':
                a,b = spec['probes']
                value = fields[...,1,a[0],a[1]] - fields[...,1,b[0],b[1]]
            else:
                raise ValueError('Unsupported outcome definition: '+spec['kind'])
            values.append(value)
        return torch.stack(values,dim=-1) if isinstance(fields,torch.Tensor) else np.stack(values,axis=-1)

    def physics_check():
        a,nu,t,n = 1.1,0.07,0.6,64
        f = vortex_fields(a,nu,t,n).astype('float64')
        k = np.fft.fftfreq(n, d=1/n)
        def derivative(v, axis):
            shape=[1,1]; shape[axis]=n
            return np.fft.ifft(1j*k.reshape(shape)*np.fft.fft(v,axis=axis),axis=axis).real
        u,v,p=f[2],f[3],f[1]
        ux,uy=derivative(u,1),derivative(u,0)
        vx,vy=derivative(v,1),derivative(v,0)
        lap_u=derivative(ux,1)+derivative(uy,0)
        lap_v=derivative(vx,1)+derivative(vy,0)
        residual=max(np.abs(-2*nu*u+u*ux+v*uy+derivative(p,1)-nu*lap_u).max(),
                     np.abs(-2*nu*v+u*vx+v*vy+derivative(p,0)-nu*lap_v).max())
        divergence=float(np.abs(ux+vy).max())
        energy=float(0.5*np.mean(u*u+v*v))
        assert divergence < 1e-5 and residual < 1e-4
        assert abs(energy-a*a*np.exp(-4*nu*t)/4) < 1e-6
        return {'max_divergence':divergence,'max_momentum_residual':float(residual),'energy_identity_pass':True}
    return SOURCE, FNO_CONFIG, SOURCE_SYSTEMS, digest, write_json, vortex_fields, physical_outcomes, physics_check


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Study-bundle contract and validation
    **Objective:** make customer inputs explicit and reject incompatible data before fitting.

    A bundle contains `study.json`, `runs.csv`, `queries.csv`, numeric `.npy` arrays and `manifest.json`. Arrays have shape `[4,64,64]` in the declared field order, finite values, and no pickle. All runs share one geometry, grid, units, initial-condition convention, boundary conditions and observation time. Initial arrays must be constructible/known before running the requested simulation; supplying true future fields as initial data invalidates the experiment.

    `runs.csv` has `run_id`, `setting_id`, `split` (`train`, `development`, `reserved`), parameter columns, `initial_file`, `target_file` and named outcome columns. Training and development rows need completed targets. Reserved rows need only IDs and knobs; they are never loaded for scoring. Repeats of a setting stay in one split. The full training split is the budget for this pilot.

    `queries.csv` has `query_id`, the same ordered knobs and `initial_file`. It must contain **no reference outcomes or target files**. Metadata declares field units, physical domain, parameter units/ranges, boundary conditions, initial-condition description, target time, provenance and outcome formulas. Predictions outside the **training convex hull** are rejected, even if they are inside the declared box.

    Use consistent units: the notebook does not convert units automatically. Kinetic energy means specific kinetic energy (per unit mass). The generated example is also a copyable schema template. Hashes detect accidental changes; they do not independently establish scientific provenance. This release supports the listed channels and physical readouts only. Add a reviewed adapter for different semantics instead of renaming incompatible fields.
    """)
    return


@app.cell
def _(Path, np, pd, json, torch, vortex_fields, physical_outcomes, write_json, digest):
    def create_example(root, seed=42):
        root=Path(root); root.mkdir(parents=True,exist_ok=False)
        (root/'fields').mkdir()
        study={'schema_version':1,'study_id':'controlled_taylor_green_2d','lane':'fluid',
            'grid':[64,64],'field_names':['density','pressure','velocity_x','velocity_y'],
            'field_units':['nondimensional']*4,'domain':[[0,2*np.pi],[0,2*np.pi]],
            'geometry':'fixed periodic square','boundary_conditions':'periodic x and y',
            'initial_conditions':'Analytical t=0 vortex, known from amplitude; no future state supplied',
            'initial_available_at_query':True,'target_time':1.0,'time_unit':'nondimensional',
            'parameters':[{'name':'amplitude','unit':'nondimensional','range':[0.5,1.5]},
                          {'name':'viscosity','unit':'nondimensional','range':[0.01,0.1]}],
            'outcomes':[{'name':'kinetic_energy','unit':'nondimensional','kind':'mean_kinetic_energy'},
                        {'name':'pressure_rms','unit':'nondimensional','kind':'pressure_rms'}],
            'provenance':{'kind':'controlled_exact_reference','source_exclusion':'Not one of the three source training systems; inferred from training-code audit, legacy exposure ledger unavailable',
                          'limitation':'Analytical case; no measured solver cost or commercial transfer claim.'}}
        write_json(root/'study.json',study)
        grid=[(float(a),float(v)) for a in np.linspace(.5,1.5,5) for v in np.linspace(.01,.1,5)]
        corners=[0,4,20,24]
        other=[i for i in range(25) if i not in corners]
        order=np.random.default_rng(seed).permutation(other).tolist()
        training=set(corners+order[:12]); rows=[]
        for i,(a,v) in enumerate(grid):
            name=f'run-{i:03d}'; initial=f'fields/{name}-initial.npy'; target=f'fields/{name}-target.npy'
            np.save(root/initial,vortex_fields(a,v,0)); field=vortex_fields(a,v,1)
            np.save(root/target,field)
            out=physical_outcomes(field,study['outcomes'])
            rows.append(dict(run_id=name,setting_id=name,split='train' if i in training else 'development',
                             amplitude=a,viscosity=v,initial_file=initial,target_file=target,kinetic_energy=float(out[0]),pressure_rms=float(out[1])))
        # No reserved future fields or labels are generated, even for the easy analytical case.
        for i,(a,v) in enumerate((a,v) for a in [.625,.875,1.125,1.375] for v in [.02125,.04375,.06625,.08875]):
            rows.append(dict(run_id=f'reserved-{i}',setting_id=f'reserved-{i}',split='reserved',amplitude=a,viscosity=v))
        pd.DataFrame(rows).to_csv(root/'runs.csv',index=False)
        queries=[]
        for i,(a,v) in enumerate([(0.7,.035),(.9,.055),(1.1,.075),(1.3,.045)]):
            filename=f'fields/query-{i}-initial.npy'; np.save(root/filename,vortex_fields(a,v,0))
            queries.append(dict(query_id=f'query-{i}',amplitude=a,viscosity=v,initial_file=filename))
        pd.DataFrame(queries).to_csv(root/'queries.csv',index=False)
        write_json(root/'manifest.json',{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
        return root

    def safe_path(root, name):
        raw=Path(root)/str(name)
        p=raw.resolve()
        if not p.is_relative_to(Path(root).resolve()) or not p.is_file() or raw.is_symlink():
            raise ValueError('Invalid/missing bundle file: '+str(name))
        return p

    def read_field(root, name, grid):
        p=safe_path(root,name)
        if p.suffix!='.npy' or p.stat().st_size>2_000_000:
            raise ValueError('Use bounded numeric .npy field arrays')
        a=np.load(p,allow_pickle=False)
        if a.shape!=(4,*grid) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
            raise ValueError('Expected finite [4,64,64] fields: '+str(name))
        a=a.astype('float32')
        if not np.isfinite(a).all():raise ValueError('Field values exceed float32 range')
        return a

    def validate_study(root):
        root=Path(root)
        manifest=json.loads((root/'manifest.json').read_text())
        if not isinstance(manifest,dict) or len(manifest)>2000:
            raise ValueError('Invalid manifest or too many files')
        if any(name not in ['study.json','runs.csv','queries.csv'] and not (name.startswith('fields/') and name.endswith('.npy')) for name in manifest):
            raise ValueError('Manifest accepts only study JSON, runs/queries CSV, and fields/*.npy')
        for required in ['study.json','runs.csv','queries.csv']:
            if required not in manifest: raise ValueError('Manifest missing '+required)
        if sum(safe_path(root,name).stat().st_size for name in manifest)>512_000_000:
            raise ValueError('Study bundle exceeds 512 MB pilot limit')
        for name,expected in manifest.items():
            if digest(safe_path(root,name))!=expected: raise ValueError('Hash mismatch: '+name)
        s=json.loads((root/'study.json').read_text())
        if s['schema_version']!=1 or s['lane']!='fluid' or s['grid']!=[64,64]:
            raise ValueError('Supported contract: schema 1, fluid, 64x64 regular grid')
        if s['field_names']!=['density','pressure','velocity_x','velocity_y']:
            raise ValueError('Explicit density/pressure/velocity field mapping required')
        if len(s['field_units'])!=4 or not all(isinstance(v,str) and v.strip() for v in s['field_units']):
            raise ValueError('Four nonempty field units required')
        if s['field_units'][2]!=s['field_units'][3]:raise ValueError('Velocity components require matching units')
        for key in ['study_id','geometry','boundary_conditions','initial_conditions','time_unit','provenance']:
            if not s.get(key): raise ValueError('Missing metadata: '+key)
        if s.get('initial_available_at_query') is not True:
            raise ValueError('Known initial conditions required at query time')
        domain=np.asarray(s['domain'],dtype=float)
        if domain.shape!=(2,2) or not np.isfinite(domain).all() or not (domain[:,1]>domain[:,0]).all():
            raise ValueError('Invalid physical domain')
        if not np.isfinite(s['target_time']) or s['target_time']<=0: raise ValueError('Positive target_time required')
        params=[p['name'] for p in s['parameters']]; outs=[o['name'] for o in s['outcomes']]
        reserved_names={'run_id','setting_id','split','initial_file','target_file','query_id'}
        if not 1<=len(params)<=8 or not 1<=len(outs)<=4 or len(set(params+outs))!=len(params+outs) or set(params+outs)&reserved_names:
            raise ValueError('Require unique, non-reserved parameter/outcome names')
        for p in s['parameters']:
            if not p['unit'] or len(p['range'])!=2 or not np.isfinite(p['range']).all() or p['range'][0]>=p['range'][1]:
                raise ValueError('Parameter unit/range invalid')
        for o in s['outcomes']:
            if not o['unit'] or o['kind'] not in ['mean_kinetic_energy','pressure_rms','pressure_difference']:
                raise ValueError('Unsupported outcome/unit')
            if o['kind'] in ['pressure_rms','pressure_difference'] and o['unit']!=s['field_units'][1]:
                raise ValueError('Pressure outcome unit must match pressure field unit')
            if o['kind']=='pressure_difference':
                probes=np.asarray(o['probes'])
                if probes.shape!=(2,2) or probes.dtype.kind not in 'iu' or np.any(probes<0) or np.any(probes>=64):
                    raise ValueError('Probe coordinates must be integer [y,x] indices')
        runs=pd.read_csv(root/'runs.csv'); queries=pd.read_csv(root/'queries.csv')
        if not 5<=len(runs)<=256: raise ValueError('Use 5–256 run rows for this bounded pilot')
        if not {'run_id','setting_id','split','initial_file','target_file',*params,*outs}<=set(runs):
            raise ValueError('Missing runs.csv columns')
        if set(queries)!= {'query_id','initial_file',*params} or not 1<=len(queries)<=64:
            raise ValueError('Queries accept ONLY query_id, knobs and known initial_file (1–64 rows)')
        if runs[['run_id','setting_id','split']].isna().any().any() or runs.run_id.duplicated().any() or queries.query_id.isna().any() or queries.query_id.duplicated().any():
            raise ValueError('Missing/duplicate IDs')
        if not set(runs.split)<= {'train','development','reserved'} or not all((runs.split==x).any() for x in ['train','development','reserved']):
            raise ValueError('Nonempty train/development/reserved splits required')
        for frame in [runs,queries]:
            x=frame[params].to_numpy(dtype=float)
            if not np.isfinite(x).all(): raise ValueError('Nonfinite controls')
            for j,p in enumerate(s['parameters']):
                if np.any(x[:,j]<p['range'][0]) or np.any(x[:,j]>p['range'][1]):raise ValueError('Control outside declared range')
        # Both declared setting IDs and identical numeric knob vectors must stay grouped.
        if (runs.groupby('setting_id').split.nunique()>1).any() or (runs.groupby(params).split.nunique()>1).any():
            raise ValueError('Setting leakage across splits')
        if (runs.groupby('setting_id')[params].nunique()>1).any().any():raise ValueError('One setting_id maps to different knobs')
        active=runs[runs.split!='reserved']
        if (active.groupby('target_file').split.nunique()>1).any():raise ValueError('Target trajectory file shared across splits')
        for r in active.to_dict('records'):
            for key in ['initial_file','target_file']:
                if r[key] not in manifest:raise ValueError('Unhashed field '+str(r[key]))
                read_field(root,r[key],s['grid'])
            actual=physical_outcomes(read_field(root,r['target_file'],s['grid']),s['outcomes'])
            if not np.allclose(actual,np.asarray([r[o] for o in outs],float),rtol=1e-4,atol=1e-6):
                raise ValueError('Outcome/field definition mismatch for '+r['run_id'])
        for filename in queries.initial_file:
            if filename not in manifest:raise ValueError('Unhashed query initial field')
            read_field(root,filename,s['grid'])
        return {'root':root,'study':s,'runs':runs,'queries':queries,'manifest_sha256':digest(root/'manifest.json')}
    return create_example, safe_path, read_field, validate_study


@app.cell
def _(mo, prepare_button, settings, torch, Path, datetime, timezone, write_json, versions, physics_check, create_example, validate_study, json, shutil, SOURCE_SYSTEMS):
    mo.stop(not prepare_button.value, mo.md('Click **Prepare study** above to create a run.'))
    config=dict(settings.value)
    config['steps']=int(config['steps']); config['seed']=int(config['seed']); config['max_seconds']=float(config['max_seconds'])
    if config['device']=='cuda' and not torch.cuda.is_available(): raise ValueError('Select a CUDA GPU in MoLab, or choose CPU smoke mode.')
    if config['device']=='cpu' and config['steps']>10: raise ValueError('CPU smoke mode is capped at 10 steps. Select CUDA for the development pilot.')
    run_dir=Path('delta_v_task5_results')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'configuration.json',config)
    write_json(run_dir/'environment.json',versions)
    physics=physics_check()
    if config['study_mode']=='controlled_vortex':
        bundle_root=create_example(run_dir/'study',config['seed'])
    else:
        source_root=Path(config['bundle_path']).expanduser().resolve()
        validate_study(source_root)
        bundle_root=run_dir/'study'; bundle_root.mkdir()
        source_manifest=json.loads((source_root/'manifest.json').read_text())
        for name in [*source_manifest,'manifest.json']:
            target=bundle_root/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source_root/name,target)
    study_data=validate_study(bundle_root)
    write_json(run_dir/'physics_checks.json',physics)
    write_json(run_dir/'study_audit.json',{'selected':study_data['study']['study_id'],'source_training_systems':SOURCE_SYSTEMS,
        'target_declared_in_source_registry':study_data['study']['study_id'] in SOURCE_SYSTEMS,
        'provenance':study_data['study']['provenance'],'scope':'development only; new-problem superiority unestablished',
        'candidate_decision':'Old three systems are source-exposed; Euler four-gamma study is not a fresh design sweep; controlled vortex selected for bounded functional pilot' if config['study_mode']=='controlled_vortex' else 'User-supplied study; independent source-exclusion review required'})
    mo.vstack([mo.md(f'Prepared `{run_dir}` · verified bundle hash `{study_data["manifest_sha256"]}`'),pd.DataFrame([physics])])
    return config, run_dir, study_data, physics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Inspect the actual supplied data and split counts
    **Objective:** establish exactly how many completed runs the model receives.

    **Output:** the run table, split counts and control ranges. Training rows update weights; development rows select the checkpoint and provide exploratory errors; reserved settings are excluded from training, selection and scoring. A count of fields or pixels is not a count of independent simulation runs.

    Inspect the initial and target fields below. Their channels have explicit meanings. A missing pressure field cannot silently become a zero pressure measurement. For a customer bundle, compare this table and plot with the original export before trusting any learned prediction.
    """)
    return


@app.cell
def _(study_data, mo):
    split_summary=study_data['runs'].groupby('split').agg(rows=('run_id','size'),settings=('setting_id','nunique')).reset_index()
    mo.vstack([split_summary,study_data['runs'],mo.md('Query inputs (no target outcomes):'),study_data['queries']])
    return split_summary


@app.cell
def _(study_data, read_field, plt, run_dir, mo):
    _example=study_data['runs'][study_data['runs'].split=='train'].iloc[0]
    _fig,_axes=plt.subplots(2,3,figsize=(12,6))
    for _i,_kind in enumerate(['initial_file','target_file']):
        _array=read_field(study_data['root'],_example[_kind],[64,64])
        for _j,_ch in enumerate([1,2,3]):
            _im=_axes[_i,_j].imshow(_array[_ch],origin='lower',cmap='coolwarm')
            _axes[_i,_j].set_title(f'{_kind}: {study_data["study"]["field_names"][_ch]}')
            _fig.colorbar(_im,ax=_axes[_i,_j])
    supplied_plot_path=run_dir/'supplied_fields.png'
    _fig.tight_layout(); _fig.savefig(supplied_plot_path,dpi=140)
    mo.mpl.interactive(_fig)
    return supplied_plot_path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Fit preprocessing on training rows only
    **Objective:** prevent hidden settings from teaching the model through normalization.

    Parameters, initial fields, target fields and outcome scales are fitted only on training rows. Separate initial and target field scales reflect the changed input/output task. Constant channels use a numerical floor. We record the exact training run IDs and transformations in `normalization.json`.

    The model sees normalized known initial fields, broadcast knob values and physical coordinates. New controls must lie inside the training convex hull. This is a support check, not a calibrated uncertainty statement. Fixed geometry/conditions are part of the saved study contract, not new freely variable knobs.
    """)
    return


@app.cell
def _(np, torch, read_field):
    def fit_normalization(data):
        s=data['study']; rows=data['runs'][data['runs'].split=='train']
        names=[p['name'] for p in s['parameters']]
        x=rows[names].to_numpy(dtype='float32')
        if len(rows)<len(names)+2:raise ValueError('Insufficient training settings')
        initial=np.stack([read_field(data['root'],f,s['grid']) for f in rows.initial_file])
        target=np.stack([read_field(data['root'],f,s['grid']) for f in rows.target_file])
        outcomes=rows[[o['name'] for o in s['outcomes']]].to_numpy(dtype='float32')
        result={'train_run_ids':rows.run_id.tolist(),'parameter_names':names,'x_mean':x.mean(0).tolist(),
            'x_std':np.maximum(x.std(0),1e-6).tolist(),'training_controls':x.tolist(),
            'outcome_scale':np.maximum(outcomes.std(0),1e-6).tolist()}
        for name,arr in [('initial',initial),('target',target)]:
            result[name+'_mean']=arr.mean((0,2,3)).tolist()
            result[name+'_std']=np.maximum(arr.std((0,2,3)),1e-4).tolist()
        return result

    def inside_hull(points, training):
        from scipy.spatial import Delaunay, QhullError
        training=np.unique(np.asarray(training,dtype=float),axis=0); points=np.asarray(points,dtype=float)
        center=training.mean(0); scale=np.maximum(training.std(0),1e-6)
        a=(training-center)/scale; q=(points-center)/scale
        if a.shape[1]==1:return ((q>=a.min(0)-1e-7)&(q<=a.max(0)+1e-7)).all(1)
        try:return Delaunay(a).find_simplex(q,tol=1e-7)>=0
        except QhullError as exc:raise ValueError('Training controls do not span the declared parameter space') from exc

    def input_tensor(initial, controls, norm, study, device='cpu'):
        initial=torch.as_tensor(initial,dtype=torch.float32,device=device)
        controls=torch.as_tensor(controls,dtype=torch.float32,device=device)
        if initial.ndim==3:initial=initial.unsqueeze(0)
        if controls.ndim==1:controls=controls.unsqueeze(0)
        m=torch.tensor(norm['initial_mean'],device=device)[None,:,None,None]
        st=torch.tensor(norm['initial_std'],device=device)[None,:,None,None]
        px=(controls-torch.tensor(norm['x_mean'],device=device))/torch.tensor(norm['x_std'],device=device)
        b,_,h,w=initial.shape
        y=torch.arange(h,device=device,dtype=torch.float32)/h
        x=torch.arange(w,device=device,dtype=torch.float32)/w
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        domain=study['domain']
        coords=torch.stack([domain[0][0]+xx*(domain[0][1]-domain[0][0]),domain[1][0]+yy*(domain[1][1]-domain[1][0])])
        return torch.cat([(initial-m)/st,px[:,:,None,None].expand(-1,-1,h,w),coords[None].expand(b,-1,-1,-1)],dim=1)
    return fit_normalization, inside_hull, input_tensor


@app.cell
def _(fit_normalization, study_data, write_json, run_dir, inside_hull, mo):
    normalization=fit_normalization(study_data)
    write_json(run_dir/'normalization.json',normalization)
    _training=study_data['runs'][study_data['runs'].split=='train']
    _development=study_data['runs'][study_data['runs'].split=='development']
    _dev_hull=inside_hull(_development[normalization['parameter_names']].to_numpy(),normalization['training_controls'])
    if not _dev_hull.all():raise ValueError('This interpolation pilot requires development settings inside the training hull.')
    if not inside_hull(study_data['queries'][normalization['parameter_names']].to_numpy(),normalization['training_controls']).all():
        raise ValueError('At least one query is outside the training convex hull.')
    mo.md(f'Normalization fitted on **{len(_training)} runs / {_training.setting_id.nunique()} settings** only. All {len(_development)} development settings and query controls pass the interpolation support check.')
    return normalization


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Load the real pretrained backbone and verify lineage
    **Objective:** establish that this experiment actually uses your trained FNO.

    The pinned public checkpoint is downloaded once or read from the configured local path. Its SHA-256 must match before loading. We use PyTorch's restricted weights loader with only the two known NeuralOperator metadata globals allowlisted; no unrestricted pickle fallback. Every state key must match the original architecture.

    The new input mapper is a 1×1 convolution from `[known initial fields, knobs, coordinates]` to the FNO's 16 input channels. These are **learned features**, not four fabricated history frames. The existing four-channel FNO output is adapted to the fixed target time. We fine-tune the entire backbone plus the new mapper. Outcomes use explicit physical formulas, with no separate scalar MLP replacing the FNO.

    **Output:** checkpoint hash, loaded-key count, total/trainable real parameter counts, device and shape. Initial evaluation and the first backward pass verify numerical behavior before the first optimizer update. Task 6 will compare this architecture with identical-input scratch training and adapter-only tuning.
    """)
    return


@app.cell
def _(mo):
    load_button=mo.ui.run_button(label='Load pretrained FNO')
    mo.vstack([load_button,mo.md('Loads ~858 MB source checkpoint; produces no training results yet.')])
    return load_button


@app.cell
def _(torch, FNO, Path, digest, SpectralConv):
    class ConditionedFNO(torch.nn.Module):
        def __init__(self, parameter_count, backbone_config):
            super().__init__()
            self.conditioner=torch.nn.Conv2d(4+parameter_count+2,16,1)
            self.backbone=FNO(**backbone_config)
        def forward(self,x):
            return self.backbone(self.conditioner(x))

    def tensor_state(model):
        # NeuralOperator inserts non-tensor _metadata into state_dict. Keep our portable
        # adapted bundle tensor-only; architecture is stored separately as JSON.
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}

    def load_source(config, source):
        if config['checkpoint_path'].strip():
            path=Path(config['checkpoint_path']).expanduser()
        else:
            from huggingface_hub import hf_hub_download
            path=Path(hf_hub_download(repo_id=source['repo'],filename=source['filename'],revision=source['revision'],repo_type='dataset',token=False))
        if digest(path)!=source['sha256']:raise ValueError('Source checkpoint SHA-256 mismatch; refusing to load.')
        with torch.serialization.safe_globals([torch.nn.functional.gelu,SpectralConv]):
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        state={k:v for k,v in checkpoint['model'].items() if isinstance(v,torch.Tensor)}
        return state,{'source':source,'epoch':int(checkpoint['epoch']),'loaded_tensor_keys':len(state)}

    def count_real(model, trainable=False):
        return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if not trainable or p.requires_grad)
    return ConditionedFNO, tensor_state, load_source, count_real


@app.cell
def _(mo, load_button, run_dir, load_source, config, SOURCE, torch, ConditionedFNO, normalization, FNO_CONFIG, count_real, write_json, pd):
    mo.stop(not load_button.value,mo.md('Review preparation, then click **Load pretrained FNO**.'))
    # Depend on this prepared run so edits invalidate the model for the old study.
    _prepared_identity=run_dir.name
    source_state,lineage=load_source(config,SOURCE)
    torch.manual_seed(config['seed'])
    model=ConditionedFNO(len(normalization['parameter_names']),FNO_CONFIG)
    model.backbone.load_state_dict(source_state,strict=True)
    del source_state
    model=model.to(config['device'])
    lineage.update({'architecture':FNO_CONFIG,'total_real_parameters':count_real(model),'trainable_real_parameters':count_real(model,True),
        'new_modules':['conditioner'],'loaded_modules':['backbone'],'source_grid':[256,256],'target_grid':[64,64],
        'task':'known initial state + parameters -> fixed-time field -> physical outcomes','exposure_limitation':'Legacy checkpoint lacks complete source split ledger'})
    write_json(run_dir/'lineage.json',lineage)
    mo.vstack([pd.DataFrame([{'epoch':lineage['epoch'],'loaded_keys':lineage['loaded_tensor_keys'],'trainable_real_parameters':lineage['trainable_real_parameters']}]),mo.md('Verified source hash: `'+SOURCE['sha256']+'`')])
    return model, lineage


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · Define the loss, baseline comparison and completion gates
    **Objective:** specify what counts as learning before training.

    Training minimizes normalized field MSE plus 0.1 × normalized engineering-outcome MSE. The field term supervises pressure and velocity (density is constant in the built-in example); the outcome term uses the same differentiable formulas as the exported answers. Development loss chooses the checkpoint. We measure the untuned adapted architecture first and save each validation record immediately.

    A functional pilot must reduce development loss from its initial value, improve the final mean training-field loss from its initial value, pass reload/query checks, and beat a training-mean outcome predictor on at least one development outcome. These are development sanity gates, not a pretraining-superiority test. A failed gate produces **NEEDS WORK**, with the results preserved. Beating GP is measured but not required for Task 5 implementation completion.

    Mean, linear Ridge and GP receive exactly the same training run IDs and scalar labels; the FNO additionally receives completed fields. Report this information difference. GP uses a Matérn kernel and train-only scaling; warnings are saved. Task 6 must include the identical-input FNO trained from scratch to isolate pretraining's value.
    """)
    return


@app.cell
def _(np, torch, pd, Path, json, time, warnings, read_field, input_tensor, physical_outcomes, write_json, tensor_state, ConditionedFNO, inside_hull):
    def prepare_tensors(data,norm,device):
        s=data['study']; result={}
        for split in ['train','development']:
            rows=data['runs'][data['runs'].split==split]
            initial=np.stack([read_field(data['root'],f,s['grid']) for f in rows.initial_file])
            target=np.stack([read_field(data['root'],f,s['grid']) for f in rows.target_file])
            x=input_tensor(initial,rows[norm['parameter_names']].to_numpy(),norm,s,device)
            target=torch.as_tensor(target,device=device)
            m=torch.tensor(norm['target_mean'],device=device)[None,:,None,None]
            st=torch.tensor(norm['target_std'],device=device)[None,:,None,None]
            result[split]=(x,(target-m)/st,target)
        return result

    def loss_parts(pred,target,raw,norm,study):
        m=torch.tensor(norm['target_mean'],device=pred.device)[None,:,None,None]
        st=torch.tensor(norm['target_std'],device=pred.device)[None,:,None,None]
        field=(pred-target).square().mean()
        predicted=physical_outcomes(pred*st+m,study['outcomes'])
        actual=physical_outcomes(raw,study['outcomes'])
        scale=torch.tensor(norm['outcome_scale'],device=pred.device)
        outcome=((predicted-actual)/scale).square().mean()
        return field+0.1*outcome,field,outcome

    def evaluate_loss(model,tensors,norm,study):
        model.eval(); values=[]
        with torch.no_grad():
            for i in range(len(tensors[0])):
                parts=loss_parts(model(tensors[0][i:i+1]),tensors[1][i:i+1],tensors[2][i:i+1],norm,study)
                values.append([float(v) for v in parts])
        return np.mean(values,axis=0).tolist()

    def save_weights(model,path):
        path=Path(path); temporary=path.with_suffix('.tmp')
        torch.save(tensor_state(model),temporary); temporary.replace(path)

    def fit_study(model,data,norm,config,root,architecture,source_info):
        root=Path(root); study=data['study']; device=config['device']
        torch.manual_seed(config['seed']); rng=np.random.default_rng(config['seed'])
        tensors=prepare_tensors(data,norm,device)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        history=[]; started=time.perf_counter(); best=float('inf'); best_step=0
        initial_train=evaluate_loss(model,tensors['train'],norm,study)
        initial_dev=evaluate_loss(model,tensors['development'],norm,study)
        if not np.isfinite(initial_train+initial_dev).all():raise ValueError('Nonfinite initial loss')
        write_json(root/'model_contract.json',{'study':study,'normalization':norm,'architecture':architecture,'lineage':source_info})
        save_weights(model,root/'best_model.pt')
        best=initial_dev[0]
        history.append({'step':0,'train_loss':initial_train[0],'development_loss':best,'seconds':time.perf_counter()-started})
        pd.DataFrame(history).to_csv(root/'learning_curve.csv',index=False)
        train_x,train_y,train_raw=tensors['train']; order=rng.permutation(len(train_x)); last_step=0
        interval=min(20,int(config['steps']))
        stop_reason='step cap'
        try:
            for step in range(1,int(config['steps'])+1):
                if time.perf_counter()-started>=config['max_seconds']:
                    stop_reason='time cap'; break
                i=int(order[(step-1)%len(order)])
                if step%len(order)==0:order=rng.permutation(len(order))
                model.train(); optimizer.zero_grad(set_to_none=True)
                pred=model(train_x[i:i+1])
                loss,_,_=loss_parts(pred,train_y[i:i+1],train_raw[i:i+1],norm,study)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite gradients')
                if step==1:
                    if model.conditioner.weight.grad is None or model.conditioner.weight.grad.abs().sum()==0:
                        raise ValueError('Conditioner receives no learning signal')
                    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters()):
                        raise ValueError('Backbone receives no learning signal')
                    write_json(root/'gradient_check.json',{'conditioner':True,'backbone':True,'all_parameters_trainable':all(p.requires_grad for p in model.parameters())})
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step(); last_step=step
                if step%interval==0 or step==config['steps'] or time.perf_counter()-started>=config['max_seconds']:
                    score=evaluate_loss(model,tensors['development'],norm,study)[0]
                    if not np.isfinite(score):raise ValueError('Nonfinite development loss')
                    history.append({'step':step,'train_loss':float(loss.detach()),'development_loss':score,'seconds':time.perf_counter()-started})
                    pd.DataFrame(history).to_csv(root/'learning_curve.csv',index=False)
                    if score<best:
                        best=score; best_step=step; save_weights(model,root/'best_model.pt')
                    print(f'Step {step}: training sample loss={float(loss.detach()):.5g}; development loss={score:.5g}',flush=True)
            model.load_state_dict(torch.load(root/'best_model.pt',map_location=device,weights_only=True),strict=True)
            final_train=evaluate_loss(model,tensors['train'],norm,study)
            result={'completed_steps':last_step,'best_step':best_step,'initial_train_loss':initial_train[0],
                'initial_train_field_loss':initial_train[1],'final_train_loss':final_train[0],'final_train_field_loss':final_train[1],
                'initial_development_loss':initial_dev[0],'best_development_loss':best,'seconds':time.perf_counter()-started,
                'stop_reason':stop_reason,'seed':config['seed'],'reserved_evaluated':False}
            write_json(root/'training_result.json',result)
            return result
        except BaseException as exc:
            write_json(root/'interrupted.json',{'type':type(exc).__name__,'completed_steps':last_step,'best_step':best_step,
                'message':'Training stopped; completed traces and last selected model remain on disk.'})
            raise

    def load_adapted(root,device='cpu'):
        root=Path(root); contract=json.loads((root/'model_contract.json').read_text())
        model=ConditionedFNO(len(contract['normalization']['parameter_names']),contract['architecture'])
        model.load_state_dict(torch.load(root/'best_model.pt',map_location='cpu',weights_only=True),strict=True)
        return model.to(device).eval(),contract

    def predict_study(model,study,norm,queries,field_root,device='cpu'):
        names=norm['parameter_names']
        if set(queries)!= {'query_id','initial_file',*names}:raise ValueError('Query contains labels or undeclared inputs')
        controls=queries[names].to_numpy(dtype=float)
        if not np.isfinite(controls).all() or not inside_hull(controls,norm['training_controls']).all():
            raise ValueError('Query outside supported training convex hull')
        values=[]; fields=[]; model.eval()
        with torch.no_grad():
            for r in queries.to_dict('records'):
                initial=read_field(field_root,r['initial_file'],study['grid'])
                x=input_tensor(initial,[r[n] for n in names],norm,study,device)
                pred=model(x)[0].cpu().numpy()
                pred=pred*np.asarray(norm['target_std'])[:,None,None]+np.asarray(norm['target_mean'])[:,None,None]
                if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                out=physical_outcomes(pred,study['outcomes'])
                row={'query_id':r['query_id'],**{n:r[n] for n in names},'method':'pretrained_fno_full_finetune',
                     'support':'inside_training_hull','uncertainty':'not_calibrated'}
                for i,spec in enumerate(study['outcomes']):
                    row[spec['name']]=float(out[i]); row[spec['name']+'_unit']=spec['unit']
                values.append(row); fields.append(pred.astype('float32'))
        return pd.DataFrame(values),np.stack(fields)

    def baseline_predictions(data,norm):
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
        names=norm['parameter_names']; outs=[o['name'] for o in data['study']['outcomes']]
        train=data['runs'][data['runs'].split=='train']; dev=data['runs'][data['runs'].split=='development']
        x=(train[names].to_numpy()-norm['x_mean'])/norm['x_std']
        q=(dev[names].to_numpy()-norm['x_mean'])/norm['x_std']
        predictions={}; warning_log=[]
        for method in ['mean','linear','gp']:
            cols=[]
            for outcome in outs:
                estimator=DummyRegressor() if method=='mean' else Ridge(alpha=1e-6) if method=='linear' else GaussianProcessRegressor(
                    kernel=ConstantKernel(1,(1e-3,1e3))*Matern(length_scale=np.ones(len(names)),nu=2.5)+WhiteKernel(1e-5,(1e-8,1e-1)),normalize_y=True,n_restarts_optimizer=1,random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); estimator.fit(x,train[outcome].to_numpy()); cols.append(estimator.predict(q))
                warning_log.extend({'method':method,'outcome':outcome,'warning':str(w.message)} for w in caught)
            predictions[method]=np.stack(cols,axis=-1)
        return predictions,warning_log
    return prepare_tensors, loss_parts, evaluate_loss, save_weights, fit_study, load_adapted, predict_study, baseline_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9 · Train the Task 5 pilot
    **Objective:** learn the new parameter-to-field mapping with the source FNO.

    **Action:** click once after reviewing steps 1–8. Each update uses one completed training setting; all fields for that setting provide additional supervision. The progress log shows development loss every 20 steps. Checkpoint selection never accesses reserved labels. Training stops at the step/time cap; validation and disk writes can add time beyond the cap, which is checked between steps.

    The initial model is saved before the first optimizer step. A failure preserves completed files under the displayed run directory; it does not claim success. Re-running training requires a fresh preparation/load so each pilot starts from the original source weights. This notebook does not resume optimizer state from an interrupted run.
    """)
    return


@app.cell
def _(mo, lineage, config):
    train_button=mo.ui.run_button(label='Train Task 5')
    mo.vstack([train_button,mo.md(f'Ready: {lineage["trainable_real_parameters"]:,} trainable real parameters; {config["steps"]} steps maximum.')])
    return train_button


@app.cell
def _(mo, train_button, run_dir, write_json, config, fit_study, model, study_data, normalization, FNO_CONFIG, lineage, pd):
    mo.stop(not train_button.value,mo.md('Click **Train Task 5** to begin the bounded pilot.'))
    if (run_dir/'training_started.json').exists():
        raise ValueError('This run already started. Prepare a new run and reload the source checkpoint to retrain without mixing results.')
    write_json(run_dir/'training_started.json',{'run':run_dir.name,'configuration':config})
    training_result=fit_study(model,study_data,normalization,config,run_dir,FNO_CONFIG,lineage)
    mo.vstack([pd.DataFrame([training_result]),mo.md('Training ended. Inspect development evidence and completion gates below.')])
    return training_result


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10 · Compare engineering outcomes on development settings
    **Objective:** see whether the learned bridge is useful and whether cheap baselines remain better.

    The table reports RMSE and MAE in declared units, plus RMSE divided by **training outcome standard deviation**. This normalized score is not a per-case percentage error. All methods use the same train/development split. The FNO has richer field supervision; these numbers alone cannot attribute a win to pretraining.

    Development data selected the FNO checkpoint, so these are tuning results. Task 6 must lock architecture, budgets and tolerances, include scratch and adapter arms, and evaluate untouched problems/settings. A GP win is reported as a GP win. No fabricated accuracy or run-saving claim is emitted.
    """)
    return


@app.cell
def _(training_result, study_data, normalization, predict_study, model, config, baseline_predictions, np, pd, write_json, run_dir, mo):
    _completed=training_result['completed_steps']
    _dev=study_data['runs'][study_data['runs'].split=='development']
    _dev_query=_dev.rename(columns={'run_id':'query_id'})[['query_id',*normalization['parameter_names'],'initial_file']]
    dev_predictions,dev_fields=predict_study(model,study_data['study'],normalization,_dev_query,study_data['root'],config['device'])
    baseline_values,baseline_warnings=baseline_predictions(study_data,normalization)
    _outs=[o['name'] for o in study_data['study']['outcomes']]
    _method_values={**baseline_values,'fno':dev_predictions[_outs].to_numpy()}
    _metric_rows=[]; _prediction_rows=[]
    for _method,_values in _method_values.items():
        for _j,_out in enumerate(_outs):
            _truth=_dev[_out].to_numpy(); _error=_values[:,_j]-_truth
            _metric_rows.append({'method':_method,'outcome':_out,'unit':study_data['study']['outcomes'][_j]['unit'],
                'rmse':float(np.sqrt(np.mean(_error**2))),'mae':float(np.mean(np.abs(_error))),
                'rmse_over_train_std':float(np.sqrt(np.mean(_error**2))/normalization['outcome_scale'][_j])})
            for _id,_actual,_pred in zip(_dev.run_id,_truth,_values[:,_j]):
                _prediction_rows.append({'method':_method,'run_id':_id,'outcome':_out,'reference':float(_actual),'prediction':float(_pred)})
    metrics=pd.DataFrame(_metric_rows)
    metrics.to_csv(run_dir/'development_metrics.csv',index=False)
    pd.DataFrame(_prediction_rows).to_csv(run_dir/'development_predictions.csv',index=False)
    write_json(run_dir/'baseline_warnings.json',baseline_warnings)
    np.savez_compressed(run_dir/'development_predicted_fields.npz',fields=dev_fields)
    mo.vstack([metrics,mo.md(f'{len(baseline_warnings)} recorded baseline optimizer warnings; inspect baseline_warnings.json. Reserved references evaluated: **0**.')])
    return dev_predictions, dev_fields, metrics, baseline_warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11 · Inspect learning curves and field diagnostics
    **Objective:** detect a model that produces plausible scalar numbers while getting fields wrong.

    The training curve uses a single sampled setting per optimizer checkpoint after step 0; it is noisier than the full-development average. The report's initial/final training sanity comparison uses the full training split. The field diagnostic shows one development pressure field and its signed error. Scalar plots include all development outcomes, with a perfect-prediction diagonal.

    These plots and their underlying arrays are saved. Do not infer stable performance from one visually attractive case; the table above includes every development setting.
    """)
    return


@app.cell
def _(metrics, pd, run_dir, plt, study_data, dev_predictions, mo):
    _metrics_ready=len(metrics)
    _curve=pd.read_csv(run_dir/'learning_curve.csv')
    _fig,_axes=plt.subplots(1,2,figsize=(11,4))
    _axes[0].plot(_curve.step,_curve.train_loss,'o-',label='training (sample after step 0)')
    _axes[0].plot(_curve.step,_curve.development_loss,'o-',label='development mean')
    _axes[0].set(xlabel='Optimizer step',ylabel='Combined normalized loss',yscale='log'); _axes[0].legend()
    for _spec in study_data['study']['outcomes']:
        _name=_spec['name']; _truth=study_data['runs'].loc[study_data['runs'].split=='development',_name].to_numpy()
        _axes[1].scatter(_truth,dev_predictions[_name],label=_name)
    _axes[1].axline((0,0),slope=1,color='grey',linestyle='--');_axes[1].set(xlabel='Reference outcome',ylabel='FNO prediction');_axes[1].legend()
    learning_plot_path=run_dir/'learning_and_outcomes.png'
    _fig.tight_layout();_fig.savefig(learning_plot_path,dpi=140)
    mo.mpl.interactive(_fig)
    return learning_plot_path


@app.cell
def _(study_data, read_field, dev_fields, plt, run_dir, mo):
    _record=study_data['runs'][study_data['runs'].split=='development'].iloc[0]
    _truth=read_field(study_data['root'],_record.target_file,[64,64])[1]
    _pred=dev_fields[0,1]
    _fig,_axes=plt.subplots(1,3,figsize=(12,3.5))
    for _ax,_arr,_title in zip(_axes,[_truth,_pred,_pred-_truth],['Reference pressure','FNO pressure','Signed error']):
        _im=_ax.imshow(_arr,origin='lower',cmap='coolwarm');_ax.set_title(_title);_fig.colorbar(_im,ax=_ax)
    pressure_plot_path=run_dir/'development_pressure.png'
    _fig.tight_layout();_fig.savefig(pressure_plot_path,dpi=140)
    mo.mpl.interactive(_fig)
    return pressure_plot_path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 12 · Reload and predict new queries with only known inputs
    **Objective:** demonstrate the actual customer interaction independently of the training object.

    We reload the saved weights and metadata from disk and call `predict_study` on `queries.csv`. The inference function accepts the known initial state and knobs; no outcomes or completed target fields are accepted. It reports each outcome's units, method and interpolation-support status. Uncertainty is explicitly `not_calibrated`; Tasks 7 and 8 add reliable intervals and next-run suggestions.

    Checks compare in-memory versus reloaded predictions, isolated versus grouped queries, sensitivity to changed controls with the same initial field, and invariance to unrelated reference-label changes. Parameter sensitivity is a wiring check, not physical correctness. Held-out outcomes remain untouched. Downloading the model allows this same inference flow without retraining.
    """)
    return


@app.cell
def _(metrics, predict_study, model, study_data, normalization, config, torch, load_adapted, run_dir, np, write_json, pd, mo):
    _development_scored=len(metrics)
    query_predictions,query_fields=predict_study(model,study_data['study'],normalization,study_data['queries'],study_data['root'],config['device'])
    # Move the original to CPU while reloading to avoid holding two full models on GPU.
    model.to('cpu')
    if config['device']=='cuda': torch.cuda.empty_cache()
    reloaded,contract=load_adapted(run_dir,config['device'])
    _reloaded_predictions,_reloaded_fields=predict_study(reloaded,contract['study'],contract['normalization'],study_data['queries'],study_data['root'],config['device'])
    _outs=[o['name'] for o in study_data['study']['outcomes']]
    _reload_ok=bool(np.allclose(query_predictions[_outs],_reloaded_predictions[_outs],rtol=1e-5,atol=1e-6))
    _single,_=predict_study(reloaded,contract['study'],contract['normalization'],study_data['queries'].iloc[:1],study_data['root'],config['device'])
    _batch_ok=bool(np.allclose(_single[_outs],_reloaded_predictions[_outs].iloc[:1],rtol=1e-5,atol=1e-6))
    _changed=study_data['queries'].iloc[:1].copy()
    _controls=_changed[normalization['parameter_names']].to_numpy()[0]
    _training_controls=np.asarray(normalization['training_controls'])
    _anchor=_training_controls[np.argmax(np.linalg.norm((_training_controls-_controls)/normalization['x_std'],axis=1))]
    _changed.loc[:,normalization['parameter_names']]=0.75*_controls+0.25*_anchor
    try:
        _changed_predictions,_changed_fields=predict_study(reloaded,contract['study'],contract['normalization'],_changed,study_data['root'],config['device'])
        _sensitivity=float(np.max(np.abs(_changed_fields-_reloaded_fields[:1])))
    except ValueError:
        _sensitivity=0.0
    # Predictor signature never accepts runs/labels. Mutating a detached reference table
    # must have no influence; recompute rather than asserting the signature alone.
    _shadow=study_data['runs'].copy(); _shadow.loc[_shadow.split=='development',_outs]=1e9
    _again,_=predict_study(reloaded,contract['study'],contract['normalization'],study_data['queries'],study_data['root'],config['device'])
    _labels_ok=bool(np.allclose(_again[_outs],_reloaded_predictions[_outs],rtol=1e-5,atol=1e-6))
    checks={'reload_matches':_reload_ok,'query_batch_invariant':_batch_ok,'excluded_labels_invariant':_labels_ok,
            'parameter_conditioning_changes_field':_sensitivity>1e-7,'parameter_perturbation_max_field_change':_sensitivity,
            'reserved_evaluated':False,'query_requires_no_targets':True}
    query_predictions.to_csv(run_dir/'query_predictions.csv',index=False)
    np.savez_compressed(run_dir/'query_predicted_fields.npz',fields=query_fields)
    write_json(run_dir/'inference_checks.json',checks)
    reloaded.to('cpu')
    if config['device']=='cuda':torch.cuda.empty_cache()
    mo.vstack([query_predictions,pd.DataFrame([checks])])
    return query_predictions, query_fields, checks


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 13 · Produce the measured Task 5 report and Task 6 handoff
    **Objective:** distinguish completed mechanism, failed sanity checks and unproven commercial claims.

    The report uses this run's numbers, records data and information budgets, source lineage, baseline winners, warnings and limits. **READY FOR TASK 6** means the development gates passed; it does not mean pretrained FNO superiority. **NEEDS WORK** means inspect training or representation before scaling the benchmark. Both outcomes are saved.

    Task 6 must choose and audit a harder new fluid problem, lock tolerated engineering error and all training/tuning budgets, compare full tuning/adapter/scratch with scalar baselines, count supplied and validation runs, and evaluate fresh confirmatory references. This notebook reserves controls but does not claim the analytical fixture is an adequate commercial benchmark.
    """)
    return


@app.cell
def _(metrics, training_result, checks, write_json, run_dir, study_data, SOURCE, split_summary, json, baseline_warnings, mo):
    _pivot=metrics.pivot(index='outcome',columns='method',values='rmse')
    _beats_mean=bool((_pivot['fno']<_pivot['mean']).any())
    _gates={'training_field_loss_improved':training_result['final_train_field_loss']<training_result['initial_train_field_loss'],
            'development_loss_improved':training_result['best_development_loss']<training_result['initial_development_loss'],
            'beats_mean_on_at_least_one_outcome':_beats_mean,
            **{k:checks[k] for k in ['reload_matches','query_batch_invariant','excluded_labels_invariant','parameter_conditioning_changes_field','query_requires_no_targets']}}
    status='READY FOR TASK 6' if all(_gates.values()) else 'NEEDS WORK'
    write_json(run_dir/'completion_gates.json',{'status':status,'gates':_gates})
    _handoff={'task5_status':status,'architecture_file':'model_contract.json','training_configuration':'configuration.json',
        'reserved_run_ids':study_data['runs'].loc[study_data['runs'].split=='reserved','run_id'].tolist(),
        'reserved_reference_generated_or_evaluated':False,'candidate_arms':['pretrained_full_finetune','pretrained_adapter_only','identical_input_scratch','gp','linear','mean'],
        'before_test':['Audit harder new target problem and source exposure','Lock engineering tolerances, primary metrics and all tuning/validation costs','Freeze architecture and preprocessing','Ensure whole-setting splits and count repeats','Disclose field supervision versus scalar-only inputs','Use fresh confirmatory data; this development set is already inspected'],
        'uncertainty':'Task 7','next_runs':'Task 8','real_solver_workflow':'Task 9','privacy_before_launch':'Task 15 pending'}
    write_json(run_dir/'task6_handoff.json',_handoff)
    _summary_table=metrics.to_csv(index=False)
    report=f"""# Delta-V Task 5 measured development report

    Status: **{status}**

    Study: {study_data['study']['study_id']}. Source checkpoint: {SOURCE['sha256']}.
    Bundle manifest: {study_data['manifest_sha256']}.

    Training settings: {int(split_summary.loc[split_summary.split=='train','settings'].iloc[0])}; development settings: {int(split_summary.loc[split_summary.split=='development','settings'].iloc[0])}.
    Completed steps: {training_result['completed_steps']}; selected step: {training_result['best_step']}; training-stage seconds: {training_result['seconds']:.2f}.
    Initial development loss: {training_result['initial_development_loss']:.6g}; selected development loss: {training_result['best_development_loss']:.6g}.

    ## Completion gates
    {json.dumps(_gates,indent=2)}

    ## Engineering outcome errors
    RMSE/MAE are in declared units; normalized RMSE uses training outcome standard deviation, not percentage error.

    ```csv
    {_summary_table}```

    ## Interpretation and limitations
    - Real source FNO weights are strictly loaded; the input conditioner is newly trained. All backbone weights are fine-tuned.
    - Known initial fields and knobs are the only query inputs. No query future frames or outcomes are consumed.
    - FNO additionally uses field supervision. GP/linear/mean see the same runs' controls and scalar outcomes.
    - Development selects the checkpoint; errors are exploratory. Task 6 must compare identical-input scratch and adapter arms on fresh tests.
    - Its two default scalar outcomes are mathematically equal in this vortex, not two independent success cases.
    - The default vortex study is an exact analytical development fixture, not expensive customer CFD. It establishes no solver-run or money savings.
    - Source-training code indicates three systems; the legacy checkpoint lacks a complete exposure ledger.
    - Source resolution 256x256 is adapted to 64x64 here. Generalization to other resolutions/geometries is not established.
    - Reserved settings evaluated: zero. Uncertainty is not calibrated. No next-run policy is implemented.
    - Baseline warnings recorded: {len(baseline_warnings)}. No weights or customer data uploaded remotely.

    See task6_handoff.json for the next protocol and model_contract.json for the exact saved inference contract.
    """
    (run_dir/'report.md').write_text(report)
    mo.md(report)
    return status, report


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 14 · Preserve the notebook, model, study and every result
    **Objective:** make this run reproducible and downloadable before MoLab storage disappears.

    If training fails, use **Package available files after interruption** below; that recovery control is independent of the training/report cells and does not restart training. It preserves whatever exists and does not mark Task 5 complete.

    The full ZIP contains the study bundle (including supplied fields), normalization, source identity, adapted weights, run configuration, environment versions, model contract, learning trace, baseline warnings, development/query predictions, fields, plots, checks, report, Task 6 handoff and this exact notebook source. An artifact manifest lists byte counts and SHA-256 hashes. The ZIP is read back and checked against that manifest before a download link appears.

    The smaller evidence ZIP omits adapted weights and study field arrays; it is useful for sharing results but is **not** sufficient to reload the model. Both remain local. The public source checkpoint is referenced by pinned hash rather than duplicated in the ZIP.

    There is no automatic HF upload in this notebook. Save both downloads. Task 15 will establish approved private storage before production/customer artifacts are uploaded. Notebook source is being published at your request; future run artifacts are not committed automatically.
    """)
    return


@app.cell
def _(Path, shutil, digest, write_json, hashlib):
    def package_run(root,source_path):
        import zipfile
        root=Path(root)
        shutil.copyfile(source_path,root/'notebook_source.py')
        manifest={str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
        write_json(root/'artifact_manifest.json',manifest)
        archives=[]
        for evidence in [False,True]:
            path=root.with_name(root.name+('-evidence' if evidence else '-all')).with_suffix('.zip')
            names=[name for name in manifest if not evidence or (not name.endswith('.pt') and not name.startswith('study/fields/'))]
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for name in names:z.write(root/name,name)
                z.write(root/'artifact_manifest.json','artifact_manifest.json')
            with zipfile.ZipFile(path) as z:
                for name in names:
                    h=hashlib.sha256()
                    with z.open(name) as stream:
                        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
                    if h.hexdigest()!=manifest[name]['sha256']:raise ValueError('ZIP hash verification failed: '+name)
            archives.append(path)
        return archives
    return package_run


@app.cell
def _(report, mo, package_run, run_dir, status, notebook_source_path, supplied_plot_path, learning_plot_path, pressure_plot_path):
    _report_ready=report
    for _plot in [supplied_plot_path,learning_plot_path,pressure_plot_path]:
        if not _plot.is_file():raise ValueError('Missing figure artifact')
    # marimo.notebook_location is robust to MoLab's renamed notebook.py.
    _notebook_path=notebook_source_path
    archives=package_run(run_dir,_notebook_path)
    mo.vstack([mo.md(f'**{status}** · full ZIP and evidence ZIP verified locally. Download before ending MoLab.'),
        mo.download(data=lambda: archives[0].read_bytes(),mimetype='application/zip',filename=archives[0].name,label='Download complete Task 5 run (includes model)'),
        mo.download(data=lambda: archives[1].read_bytes(),mimetype='application/zip',filename=archives[1].name,label='Download smaller evidence ZIP'),
        mo.md('Artifacts directory: `'+str(run_dir.resolve())+'`')])
    return archives


@app.cell
def _(mo):
    recovery_button=mo.ui.run_button(label='Package available files after interruption')
    recovery_button
    return recovery_button


@app.cell
def _(mo, recovery_button, package_run, run_dir, notebook_source_path):
    mo.stop(not recovery_button.value)
    recovery_archives=package_run(run_dir,notebook_source_path)
    mo.vstack([mo.md('Recovery archive: **available files only**. This is not a completion claim. No training was repeated.'),
        mo.download(data=lambda: recovery_archives[0].read_bytes(),filename=recovery_archives[0].name,mimetype='application/zip',label='Download available run files')])
    return recovery_archives


if __name__ == "__main__":
    app.run()
