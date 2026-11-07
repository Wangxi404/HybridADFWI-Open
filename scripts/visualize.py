#!/usr/bin/env python3
"""Rebuild the scientific gallery from real Marmousi inputs and archived evidence.

Prepare paper + smoke models first. A local CPU native run supplies wavefields
and receiver data; archived GPU timings are never rerun by this script.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts' / 'figure_qa'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'lines.linewidth': 1.1,
    'savefig.dpi': 300,
})
BLUE, ORANGE = '#37679d', '#c56c34'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def model_load(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Prepare the real model first: {path.name}')
    with np.load(path, allow_pickle=False) as f:
        return {k: v.copy() for k, v in f.items()}


def summary_stats(a):
    a = np.asarray(a)
    return {'shape': list(a.shape), 'min': float(a.min()), 'max': float(a.max()),
            'l2': float(np.linalg.norm(a)), 'finite': bool(np.isfinite(a).all())}


def symmetric_scale(arrays, percentile=99.5):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    limit = float(np.percentile(np.abs(values), percentile))
    if limit == 0:
        limit = float(np.max(np.abs(values))) or 1.
    return limit, {'absolute_percentile': percentile, 'limit': limit,
                   'clipped_fraction': [float(np.mean(np.abs(a) > limit)) for a in arrays],
                   'transform': 'display saturation only; no array modification'}


def plate(nrows, ncols, size, title, subtitle, left=.075, right=.92, bottom=.12, top=.82,
          wspace=.28, hspace=.62):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)
    fig.text(left, .965, title, ha='left', va='top', fontsize=12, weight='bold')
    fig.text(left, .91, subtitle, ha='left', va='top', fontsize=8, color='#505050')
    for label, ax in zip(string.ascii_lowercase, axes.flat):
        ax.set_gid(label)
        ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-18, 9),
                    textcoords='offset points', ha='left', va='bottom', fontsize=10, weight='bold')
    return fig, axes


def image(ax, a, extent, **kwargs):
    return ax.imshow(a, extent=extent, origin='upper', interpolation='nearest', aspect='auto', **kwargs)


def map_extent(model):
    dx, dz = float(model['dx']), float(model['dz'])
    return [(float(model['x'][0])-dx/2)/1000, (float(model['x'][-1])+dx/2)/1000,
            (float(model['z'][-1])+dz/2)/1000, (float(model['z'][0])-dz/2)/1000]


def decorate_map(ax, bottom=True, left=True):
    ax.set_xticks([5, 7, 9, 11, 13])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel('Distance x (km)' if bottom else '')
    ax.set_ylabel('Depth z (km)' if left else '')


def export(fig, name, out, qa, axes):
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
        json_out=qa / f'{name}.alignment.json', overlay_svg=qa / f'{name}.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(out / f'{name}.png', dpi=300, facecolor='white')
    fig.savefig(out / f'{name}.pdf', facecolor='white')
    fig.savefig(out / f'{name}.svg', facecolor='white')
    plt.close(fig)


def capture_cpu(smoke_path, cache_path, refresh=False):
    if cache_path.is_file() and not refresh:
        cached = model_load(cache_path)
        meta = json.loads(str(cached['metadata_json']))
        if meta['model_sha256'] == sha(smoke_path) and meta.get('capture_version') == 1:
            return cached
    import torch
    from hybridadfwi.experiment import build, objective, verify_upstream, UPSTREAM_COMMIT
    from hybridadfwi.adjoint import hybrid_forward, preserve_padding_dtype, disable_native_checkpointing
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    verify_upstream()
    cfg = dict(model=str(smoke_path), device='cpu', nt=640, dt=.004, f0=5.,
               shots=1, receivers=25, depth_m=540., nabc=10, free_surface=True,
               fd_order=8, dtype='float64', parameters=['vp', 'vs', 'rho'],
               checkpointing=False, segments=1, threads=1, objective='l2')
    print('Capturing actual native Marmousi-II CPU wavefields (640 steps)...', flush=True)
    true_model, true_prop = build(cfg, truth=True)
    observed, tape = hybrid_forward(true_prop, checkpointing=False, checkpoint_segments=1)
    # Native staggered v_z is recorded before each velocity update, as used by
    # the exact reverse. Crop only the PML and four top ghost rows.
    nz, nx = true_model.nz, true_model.nx
    frames_idx = np.arange(0, cfg['nt'], 8)
    vz = np.stack([tape.full_history[int(i)][1][0, 4:4+nz, 10:10+nx].cpu().numpy().copy()
                   for i in frames_idx])
    receiver = observed['vz'].cpu().numpy().copy()
    instrumented_digest = tape.stats['instrumented_native_source_sha256']
    tape.full_history.clear()
    del tape, observed, true_model, true_prop
    gc.collect()
    # Validate the instrumented receiver trace against the pinned native loop.
    with torch.no_grad():
        _, native_prop = build(cfg, truth=True, trainable=False)
        with preserve_padding_dtype(), disable_native_checkpointing():
            native_receiver = native_prop.forward(fd_order=8, checkpoint_segments=1)['vz']
    bitwise = bool(np.array_equal(receiver, native_receiver.numpy()))
    if not bitwise:
        raise AssertionError('History capture changed native receiver samples')
    del native_prop, native_receiver
    _, initial_prop = build(cfg)
    prediction, initial_tape = hybrid_forward(initial_prop, checkpointing=False, checkpoint_segments=1)
    data = prediction['vz'].detach().requires_grad_(True)
    observation = torch.from_numpy(receiver)
    scale = max(float(observation.square().mean().sqrt()), 1e-30)
    loss = objective(data, observation, scale, cfg['dt'], cfg['objective'])
    source, = torch.autograd.grad(loss, data)
    prediction_np = prediction['vz'].cpu().numpy().copy()
    residual = prediction_np - receiver
    metadata = dict(capture_version=1, config={k: v for k, v in cfg.items() if k != 'model'},
                    model='data/prepared/marmousi2_smoke.npz', model_sha256=sha(smoke_path),
                    upstream_commit=UPSTREAM_COMMIT,
                    instrumented_native_source_sha256=instrumented_digest,
                    native_receiver_bitwise_equal=bitwise, observed_rms=scale, objective=float(loss),
                    sample_definition='v_z before velocity update at native loop index i; t=i*dt',
                    field_crop='remove top 4 ghost rows and left/right 10-cell PML; retain physical 39x101 grid',
                    source_definition='torch.autograd.grad(0.5*mean(((pred-obs)/observed_RMS)**2), pred)',
                    velocity_location='native staggered-grid component; no collocation interpolation',
                    source_amplitude='pinned native Ricker moment-source convention; no absolute calibration',
                    runtime=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vz_frames=vz, frame_indices=frames_idx,
                        observed=receiver, predicted=prediction_np, residual=residual,
                        adjoint_source=source.detach().numpy(), metadata_json=np.array(json.dumps(metadata)))
    initial_tape.full_history.clear()
    print('Native receiver traces are bitwise equal; CPU capture saved.', flush=True)
    return model_load(cache_path)


def model_figure(model, out, qa, manifest):
    extent = map_extent(model)
    fig, axs = plate(2, 3, (10.8, 5.7), 'Shared model: true structure and smoothed starting point',
        'Marmousi-II · 78 × 200 cells · 45 m grid · Gaussian σ = 150 m in solids; true water restored',
        right=.88, bottom=.14, hspace=.7)
    sources = {}
    for c, (p, label, unit, factor) in enumerate([
        ('vp', 'Vp', 'km/s', .001), ('vs', 'Vs', 'km/s', .001), ('rho', 'Density', 'g/cm³', .001)]):
        arrays = [model[f'{p}_{s}'] * factor for s in ('true', 'initial')]
        lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
        # A nonzero scale is needed for the constant density control.
        if hi == lo:
            lo, hi = lo-.1, hi+.1
        norm = Normalize(lo, hi)
        for r, a in enumerate(arrays):
            im = image(axs[r,c], a, extent, cmap='viridis', norm=norm)
            axs[r,c].set_title(f'{label} · {"True" if r == 0 else "Initial"}', loc='left', pad=10)
            decorate_map(axs[r,c], bottom=r == 1, left=c == 0)
        x = axs[1,c].get_position().x0
        cbax = fig.add_axes([x, .058, axs[1,c].get_position().width, .018])
        cb = fig.colorbar(im, cax=cbax, orientation='horizontal')
        cb.set_label(unit, labelpad=1)
        cb.ax.tick_params(labelsize=7, pad=1)
        if p == 'rho':
            cb.set_ticks([2.35, 2.45, 2.55])
        sources[p] = {'true': summary_stats(model[p+'_true']), 'initial': summary_stats(model[p+'_initial']),
                      'display_units': unit, 'display_scale': [float(lo), float(hi)], 'clipped_fraction': 0.}
    export(fig, 'models', out, qa, axs)
    manifest['figures']['models'] = {'data': 'data/prepared/marmousi2_paper.npz', 'fields': sources,
                                     'note': 'rho is the actual constant benchmark control, not a recovered density map'}


def kernel_figure(model, out, qa, manifest):
    directory = ROOT / 'evidence' / 'marmousi2-fd8-f64-20261106'
    ad, hy = model_load(directory / 'autograd-0.kernels.npz'), model_load(directory / 'hybrid-0.kernels.npz')
    extent = map_extent(model)
    fig, axs = plate(3, 3, (11.6, 8.), 'Same discrete gradient, checked against native AD',
        '9 shots · 2500 steps · 50 checkpoints · repeat 0 · AD/Hybrid: shared symlog scale · difference: linear scale',
        right=.88, bottom=.14, top=.84, wspace=.42, hspace=.73)
    records = {}
    for r, p in enumerate(('vp','vs','rho')):
        a, h = ad['gradient_'+p], hy['gradient_'+p]
        limit, clipping = symmetric_scale([a,h])
        peak = float(np.max(np.abs(a)))
        difference = (h-a)/peak
        dlim = float(np.max(np.abs(difference))) or np.finfo(float).eps
        error = float(np.linalg.norm(h-a)/np.linalg.norm(a))
        for c, (arr, lim) in enumerate(((a,limit),(h,limit),(difference,dlim))):
            im = image(axs[r,c], arr, extent, cmap='RdBu_r',
                norm=SymLogNorm(linthresh=lim*.001, vmin=-lim, vmax=lim, base=10) if c < 2 else Normalize(-lim,lim))
            title = ('Native AD', 'Hybrid', f'(Hybrid − AD) / max |AD|\nrelative L2 = {error:.2e}')[c]
            axs[r,c].set_title(f'{p.capitalize() if p != "rho" else "ρ"} · {title}', loc='left', pad=10)
            decorate_map(axs[r,c], bottom=r == 2, left=c == 0)
            if c in (1,2):
                pos=axs[r,c].get_position()
                cbax=fig.add_axes([pos.x1+.008, pos.y0, .009, pos.height])
                cb=fig.colorbar(im, cax=cbax)
                if c == 1:
                    cb.set_ticks([-lim,-lim*.01,0,lim*.01,lim])
                    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:.0e}' if v else '0'))
                cb.ax.tick_params(labelsize=7, pad=2)
                if c == 2:
                    fmt=ScalarFormatter(useMathText=False)
                    fmt.set_powerlimits((0,0))
                    cb.formatter=fmt
                    cb.update_ticks()
        clipping['color_normalization'] = 'symmetric log, base 10, linear core = display limit * 0.001'
        records[p]={'native': summary_stats(a), 'hybrid': summary_stats(h), 'relative_l2': error,
                    'shared_native_hybrid_display': clipping,
                    'difference_display': {'formula':'(hybrid-native)/max(abs(native))', 'limit':dlim,
                                           'clipped_fraction':0.},
                    'units':'derivative of fixed-RMS dimensionless objective per SI parameter unit'}
    fig.text(.075,.025,'At smoothed initial model; true-model target. AD/Hybrid: shared symlog, 0.5% display saturation, no smoothing.\nColorbars: objective derivative per SI parameter unit; linear core = 0.001 × limit. Difference: dimensionless, no clipping.',fontsize=8,color='#505050')
    export(fig, 'gradients', out, qa, axs)
    manifest['figures']['gradients']={'archive':'evidence/marmousi2-fd8-f64-20261106', 'repeat':0,
                                    'arrays':records, 'display_smoothing':'none',
                                    'evaluation_model':'prepared smoothed initial model',
                                    'target_data':'native forward observation from true model'}


def performance_figure(out, qa, manifest):
    directories=['marmousi2-fd8-f64-20261106','no-checkpoint-1shot-20261106']
    summaries=[json.loads((ROOT/'evidence'/d/'summary.json').read_text()) for d in directories]
    fig, axs=plate(2,2,(10.5,7.3),'Measured cost and agreement, with every repeat visible',
        'RTX A5000 · FD8 · float64 · 78 × 200 · 2500 steps · n = 3 paired fresh-process repeats per protocol',
        right=.94,bottom=.13,top=.83,wspace=.32,hspace=.65)
    rows=[]
    for c,(d,s) in enumerate(zip(directories,summaries)):
        ax=axs[0,c]
        title='9 shots · 50 checkpoint segments' if c == 0 else '1 shot · checkpointing OFF'
        ax.set_title(title,loc='left',pad=13)
        for j,(method,label,color) in enumerate([('autograd','Native AD',BLUE),('hybrid','Hybrid',ORANGE)]):
            ys=np.array(s['statistics'][method]['total_s'])
            ax.scatter(np.full(3,j)+np.array([-.07,0,.07]),ys,color=color,s=25,zorder=3)
            ax.errorbar(j+.17,ys.mean(),yerr=ys.std(ddof=1),fmt='s',color=color,capsize=4,markersize=5,zorder=3)
            ax.text(j, .96, f'{ys.mean():.1f} ± {ys.std(ddof=1):.1f} s',transform=ax.get_xaxis_transform(),
                    ha='center',va='top',fontsize=8,color=color)
            for repeat,y in enumerate(ys):
                rows.append({'protocol':d,'method':method,'repeat':repeat,'total_s':float(y),
                    'peak_cuda_allocated_GiB':s['statistics'][method]['peak_cuda_allocated_bytes'][repeat]/2**30,
                    **{p+'_relative_l2':s['comparison'][repeat]['kernels'][p]['relative_l2'] for p in ('vp','vs','rho')}})
        for rep in range(3):
            a=s['statistics']['autograd']['total_s'][rep]
            h=s['statistics']['hybrid']['total_s'][rep]
            ax.plot([-.07+.07*rep, .93+.07*rep],[a,h],color='#c5c5c5',lw=.7,zorder=1)
        ax.set_xlim(-.4,1.45); ax.set_ylim(0,max(s['statistics']['autograd']['total_s'])*1.25)
        ax.set_xticks([0,1],['Native AD','Hybrid']);ax.set_ylabel('Complete gradient time (s)')
        ax=axs[1,c]
        for j,(method,label,color) in enumerate([('autograd','Native AD',BLUE),('hybrid','Hybrid',ORANGE)]):
            mem=np.array(s['statistics'][method]['peak_cuda_allocated_bytes'])/2**30
            ax.scatter(np.full(3,j)+np.array([-.07,0,.07]),mem,color=color,s=25,zorder=3)
            ax.errorbar(j+.17,mem.mean(),yerr=mem.std(ddof=1),fmt='s',color=color,capsize=4,markersize=5,zorder=3)
            ax.text(j,.96,f'{mem.mean():.3f} GiB',transform=ax.get_xaxis_transform(),ha='center',va='top',color=color)
        for rep in range(3):
            a=s['statistics']['autograd']['peak_cuda_allocated_bytes'][rep]/2**30
            h=s['statistics']['hybrid']['peak_cuda_allocated_bytes'][rep]/2**30
            ax.plot([-.07+.07*rep,.93+.07*rep],[a,h],color='#c5c5c5',lw=.7,zorder=1)
        errs=[x['kernels'][p]['relative_l2'] for x in s['comparison'] for p in ('vp','vs','rho')]
        ax.set_title(f'CUDA allocated peak · max kernel error {max(errs):.2e}',loc='left',pad=13)
        ax.set_xlim(-.4,1.45); ax.set_ylim(0,max(s['statistics']['autograd']['peak_cuda_allocated_bytes'])/2**30*1.28)
        ax.set_xticks([0,1],['Native AD','Hybrid']);ax.set_ylabel('Peak allocated memory (GiB)')
    fig.text(.075,.042,'Dots: every repeat; squares ± bars: mean ± sample SD. Memory SD is zero in these archives.\nProtocols differ in shot count and host contention; compare methods within each column.',fontsize=8,color='#505050')
    export(fig,'performance',out,qa,axs)
    with (out/'benchmark_points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest['figures']['performance']={'sources':['evidence/'+d+'/summary.json' for d in directories],
          'source_data':'docs/figures/benchmark_points.csv','n':3,'replicate':'paired fresh-process runs',
          'summary':'mean ± sample SD (ddof=1)','memory_definition':'CUDA peak allocated; not reserved or device total',
          'caveat':'checkpointed archive had CPU contention; separate shot counts prohibit cross-protocol speed claims'}


def wavefield_figure(smoke, capture, out, qa, manifest, gif=True):
    extent=map_extent(smoke)
    frames=capture['vz_frames']
    meta=json.loads(str(capture['metadata_json']))
    ids=[int(np.argmin(np.abs(capture['frame_indices']*.004-t))) for t in (.384,.64,.96,1.28,1.792,2.304)]
    raw=[frames[i] for i in ids]
    peak=float(np.max(np.abs(frames)))
    limit,clipping=symmetric_scale(raw)
    fig,axs=plate(2,3,(11.2,5.9),'A real native wavefield evolving through Marmousi-II',
        'New CPU illustration · true model · 39 × 101 cells · 90 m · 1 shot · FD8 · dt = 4 ms · float64',
        right=.88,bottom=.14,hspace=.7)
    cfg=meta['config']
    sx=int(np.rint(np.linspace(8, int(smoke['nx'])-9, cfg['shots']+2)[1]))
    rx=np.unique(np.rint(np.linspace(5,int(smoke['nx'])-6,cfg['receivers'])).astype(int))
    source_x=float(smoke['x'][sx])/1000
    receiver_x=smoke['x'][rx]/1000
    for panel,(ax,i) in enumerate(zip(axs.flat,ids)):
        im=image(ax,frames[i]/peak,extent,cmap='RdBu_r',vmin=-limit/peak,vmax=limit/peak)
        t=float(capture['frame_indices'][i])*.004
        ax.set_title(f'Native v_z · t = {t:.3f} s',loc='left',pad=10)
        decorate_map(ax,bottom=ax in axs[1],left=ax in axs[:,0])
        if panel == 0:
            # Thin outlined marks preserve the underlying signed wavefield.
            ax.scatter(receiver_x, np.full(len(rx),.540), marker='v', s=9,
                       facecolors='none', edgecolors='#6b6b6b', linewidths=.5, zorder=4)
            ax.scatter([source_x],[.540],marker='*',s=45,facecolors='#111111',edgecolors='white',linewidths=.4,zorder=5)
    cbax=fig.add_axes([.91,.27,.013,.43]);cb=fig.colorbar(im,cax=cbax)
    cb.set_label('v_z / global peak |v_z|',labelpad=7)
    fig.text(.075,.045,'Panel a: star = source; outlined triangles = 25 receivers at z = 540 m. Shared scale across all frames.\nNative staggered velocity before its update; physical domain shown, PML and top ghost cells omitted.',fontsize=8,color='#505050')
    export(fig,'wavefields',out,qa,axs)
    manifest['figures']['wavefields']={'capture':meta,'times_s':[float(capture['frame_indices'][i])*.004 for i in ids],
          'frame_display':clipping,'global_peak_over_all_80_frames':peak,'normalization':'all fields divided by same global absolute peak',
          'origin':'five-field history instrumentation of pinned native FD8 loop; receiver traces validated bitwise against native'}
    manifest['figures']['wavefields']['geometry']={'source_x_km':source_x,'receiver_x_km':receiver_x.tolist(),'depth_m':540}
    if gif:
        fig,ax=plt.subplots(figsize=(8,3.7))
        fig.subplots_adjust(left=.085,right=.83,bottom=.18,top=.84)
        # Use exactly the static-plate scale; no frame-dependent gain.
        im=image(ax,frames[0]/peak,extent,cmap='RdBu_r',vmin=-limit/peak,vmax=limit/peak)
        decorate_map(ax)
        ax.set_title('Native Marmousi-II v_z · 1 shot · t = 0.000 s',loc='left',fontsize=10)
        cbax=fig.add_axes([.88,.23,.018,.51]);fig.colorbar(im,cax=cbax,label='v_z / global peak |v_z|')
        fig.text(.085,.035,'True model · FD8 · 90 m grid · fixed color scale · CPU illustration',fontsize=8)
        def update(i):
            im.set_data(frames[i]/peak)
            ax.set_title(f'Native Marmousi-II v_z · 1 shot · t = {int(capture["frame_indices"][i])*.004:.3f} s',loc='left',fontsize=10)
            return im,
        anim=FuncAnimation(fig,update,frames=len(frames),interval=65,blit=False)
        anim.save(out/'wavefield.gif',writer=PillowWriter(fps=15),dpi=fig.dpi)
        plt.close(fig)
        manifest['figures']['wavefield_animation']={'frames':len(frames),'native_index_stride':8,'frame_dt_s':.032,
          'playback_fps':15,'display':clipping,'note':'same fixed scale as static wavefield plate; not real-time playback',
          'clipped_fraction_all_frames':float(np.mean(np.abs(frames)>limit))}


def data_figure(smoke,capture,out,qa,manifest):
    meta=json.loads(str(capture['metadata_json']))
    scale=meta['observed_rms']
    cfg=meta['config'];rx=np.unique(np.rint(np.linspace(5,101-6,cfg['receivers'])).astype(int))
    receiver_x=smoke['x'][rx]/1000
    x_edges=np.r_[receiver_x[0]-(receiver_x[1]-receiver_x[0])/2,
                  (receiver_x[:-1]+receiver_x[1:])/2,
                  receiver_x[-1]+(receiver_x[-1]-receiver_x[-2])/2]
    time_edges=(np.arange(cfg['nt']+1)-.5)*cfg['dt']
    obs,pred,res,source=[capture[k][0] for k in ('observed','predicted','residual','adjoint_source')]
    limit,shared=symmetric_scale([obs/scale,pred/scale])
    rlim,rclip=symmetric_scale([res/scale]);slim,sclip=symmetric_scale([source])
    fig,axs=plate(2,2,(9.2,7.9),'From forward data to the explicit adjoint receiver source',
        f'Same CPU shot · 25 receivers at z = 540 m · fixed observed RMS normalization · Φ = {meta["objective"]:.4f}',
        left=.09,right=.86,top=.84,bottom=.12,wspace=.48,hspace=.62)
    for ax,a,lim,title,unit in zip(axs.flat,[obs/scale,pred/scale,res/scale,source],
            [limit,limit,rlim,slim],['True-model observation','Initial-model prediction','Prediction − observation','AD source ∂Φ/∂d'],
            ['Amplitude / observed RMS','Amplitude / observed RMS','Residual / observed RMS','Reciprocal native amplitude']):
        im=ax.pcolormesh(x_edges,time_edges,a,cmap='RdBu_r',vmin=-lim,vmax=lim,
                         shading='flat',rasterized=True)
        ax.set_ylim(time_edges[-1],time_edges[0])
        ax.set_title(title,loc='left',pad=12)
        ax.set_xlabel('Receiver x (km)');ax.set_ylabel('Time (s)')
        ax.set_xticks([6,9,12]);ax.set_yticks([0,.8,1.6,2.4])
        pos=ax.get_position();cbax=fig.add_axes([pos.x1+.012,pos.y0,.012,pos.height])
        cb=fig.colorbar(im,cax=cbax);cb.ax.tick_params(labelsize=7)
        cb.set_label(unit,labelpad=5,fontsize=7)
        if title.startswith('AD'):
            fmt=ScalarFormatter(useMathText=False);fmt.set_powerlimits((0,0));cb.formatter=fmt;cb.update_ticks()
    fig.text(.09,.038,'Φ = ½ mean[((d − d_obs) / RMS_obs)²]. Source is computed by torch.autograd.grad.\nSeparate residual/source scales are labelled; all fields are raw solver samples with display saturation only.',fontsize=8,color='#505050')
    export(fig,'data_adjoint_source',out,qa,axs)
    manifest['figures']['data_adjoint_source']={'capture':meta,'shared_observation_prediction':shared,
          'residual_display':rclip,'source_display':sclip,
          'raw_fields':{k:summary_stats(capture[k]) for k in ('observed','predicted','residual','adjoint_source')},
          'source_identity_max_absolute_error':float(np.max(np.abs(source-res/(source.size*scale**2))))}


def agreement_figure(out, qa, manifest):
    directories = ['marmousi2-fd8-f64-20261106', 'no-checkpoint-1shot-20261106']
    fig, axs = plate(1, 2, (10.3, 4.4), 'Every parameter and repeat agrees with native AD',
        'Raw gradient relative L2 errors · three repeats each · receiver waveforms bitwise equal in all six pairs',
        right=.94, bottom=.2, top=.75, wspace=.32)
    for c, directory in enumerate(directories):
        summary = json.loads((ROOT/'evidence'/directory/'summary.json').read_text())
        ax = axs[0, c]
        ax.set_title('9 shots · 50 checkpoints' if c == 0 else '1 shot · checkpointing OFF', loc='left', pad=12)
        for j, p in enumerate(('vp', 'vs', 'rho')):
            errors = np.array([r['kernels'][p]['relative_l2'] for r in summary['comparison']])
            assert np.all(errors > 0), 'Log scale requires strictly positive errors'
            ax.scatter(j+np.array([-.08, 0, .08]), errors, color=BLUE, s=25, zorder=3)
            ax.text(j, errors.max()*2.6, f'{errors.max():.2e}', ha='center', va='bottom', fontsize=8)
        ax.axhline(1e-8, color='#888888', linestyle='--', linewidth=.8)
        ax.text(.0, 1.65e-8, 'Benchmark tolerance 1e−8', fontsize=8, color='#666666')
        ax.set_yscale('log'); ax.set_ylim(1e-15, 1e-7); ax.set_xlim(-.45, 2.45)
        ax.set_xticks([0, 1, 2], ['Vp', 'Vs', 'ρ']); ax.set_ylabel('Relative L2 error')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.0e}'))
    fig.text(.075, .045, 'Dots: all three paired repeats; labels: maximum over repeats for that parameter. No filtering or gradient smoothing.\nThese are derivative-agreement checks for the same forward operator, not inversion quality or spatial-convergence tests.', fontsize=8, color='#505050')
    export(fig, 'agreement', out, qa, axs)
    manifest['figures']['agreement'] = {
        'sources': ['evidence/'+d+'/summary.json' for d in directories],
        'metric': 'norm(hybrid-native)/norm(native), unmodified arrays',
        'all_repeats': True, 'all_parameters': ['vp', 'vs', 'rho'],
        'source_data': 'docs/figures/benchmark_points.csv', 'tolerance': 1e-8}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-model',type=Path,default=ROOT/'data/prepared/marmousi2_paper.npz')
    p.add_argument('--smoke-model',type=Path,default=ROOT/'data/prepared/marmousi2_smoke.npz')
    p.add_argument('--refresh-capture',action='store_true')
    p.add_argument('--no-gif',action='store_true')
    p.add_argument('--only',choices=['models','gradients','performance','agreement','wavefields','data'])
    args=p.parse_args()
    out=ROOT/'docs/figures';qa=ROOT/'docs/figure-qa'
    out.mkdir(exist_ok=True,parents=True);qa.mkdir(exist_ok=True,parents=True)
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() and args.only else {
        'schema_version':1,'generator':'scripts/visualize.py','figures':{},
        'attribution':'Marmousi-II: Allied Geophysical Laboratories, University of Houston; Gary S. Martin, Robert Wiley and Kurt J. Marfurt',
        'raw_model_distribution':'excluded; figures are research outputs; see docs/DATA_PROVENANCE.md',
        'backend':{'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':mpl.__version__},
        'rendering':{'interpolation':'nearest','gradient_smoothing':'none','gradient_errors':'computed on raw arrays before display saturation'},
        'export':{'png_dpi':300,'pdf_fonttype':42,'svg_fonttype':'none','minimum_main_font_pt':7,'scope':'repository/report figures, not resized journal submission masters'}}
    model=model_load(args.paper_model)
    paper_metadata=json.loads(str(model['metadata_json']))
    assert model['vp_true'].shape == (78,200), 'Archived kernel figure requires the paper grid'
    assert paper_metadata['initial_smoothing_sigma_m'] == 150., 'Use the archived 150 m initial smoothing'
    assert paper_metadata['density_mode'] == 'constant2450', 'Archived kernels use constant benchmark density'
    manifest['prepared_model_sha256']={'paper':sha(args.paper_model),'smoke':sha(args.smoke_model)}
    manifest['preparation']=json.loads(str(model['metadata_json']))
    for name,func in [('models',lambda:model_figure(model,out,qa,manifest)),
                      ('gradients',lambda:kernel_figure(model,out,qa,manifest)),
                      ('performance',lambda:performance_figure(out,qa,manifest)),
                      ('agreement',lambda:agreement_figure(out,qa,manifest))]:
        if args.only in (None,name):
            func();json_write(manifest_path,manifest);print('Exported',name,flush=True)
    if args.only in (None,'wavefields','data'):
        smoke=model_load(args.smoke_model)
        capture=capture_cpu(args.smoke_model,ROOT/'.cache/visualization/native_smoke.npz',args.refresh_capture)
        manifest['capture_sha256']=sha(ROOT/'.cache/visualization/native_smoke.npz')
        if args.only in (None,'wavefields'):
            wavefield_figure(smoke,capture,out,qa,manifest,not args.no_gif)
        if args.only in (None,'data'):
            data_figure(smoke,capture,out,qa,manifest)
    manifest['generator_sha256']=sha(__file__)
    json_write(manifest_path,manifest)
    print('Gallery complete:',out,flush=True)

if __name__=='__main__':
    main()
