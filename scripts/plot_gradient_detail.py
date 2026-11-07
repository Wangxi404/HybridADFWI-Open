#!/usr/bin/env python3
"""Render a checked gradient-detail run; no propagation or gradient processing.

Model inputs remain local. Raw and constrained-coordinate derivatives are
different saved outputs, not interchangeable display masks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hybrid-gradient-detail-mpl')
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
    'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
    'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, axes, output, name, bars):
    fig.canvas.draw()
    for image, ax, label, ticks in bars:
        box=ax.get_position(); height=fig.get_figheight()*72
        cax=fig.add_axes([box.x0, box.y0-35/height, box.width, 5/height])
        cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=ticks,
            format=mpl.ticker.FuncFormatter(lambda value,position:f'{value:.2g}'))
        cb.ax.tick_params(labelsize=7,pad=2)
        cb.set_label(label,fontsize=7,labelpad=3)
    fig.canvas.draw()
    axes=np.asarray(axes); labels=np.array(list('abcdefghijklmnopqrstuvwxyz')[:axes.size]).reshape(axes.shape)
    require_matplotlib_panel_alignment(fig,axes=axes.ravel().tolist(),panel_ids=labels.ravel().tolist(),
        row_groups=[row.tolist() for row in labels if len(row)>1],
        column_groups=[column.tolist() for column in labels.T if len(column)>1],
        json_out=output/'qa'/f'{name}.alignment.json',overlay_svg=output/'qa'/f'{name}.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(output/f'{name}.png',dpi=300)
    fig.savefig(output/f'{name}.pdf',dpi=300)
    fig.savefig(output/f'{name}.svg',dpi=300)
    plt.close(fig)


def decorate(ax,x,z):
    ax.set(xlim=(x[0],x[-1]),ylim=(z[-1],z[0]),xlabel='Original x (km)',ylabel='Depth (km)')
    ax.set_aspect('equal',adjustable='box')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/gradient_detail')
    parser.add_argument('--percentile',type=float,default=99.)
    parser.add_argument('--constrained-scope',action='store_true',
        help='Require independent hash-bound acceptance of declared coordinate gradients; retain a failed full-physical gate explicitly')
    args=parser.parse_args()
    if not 95<=args.percentile<=100:
        raise ValueError('Declare a conservative, reproducible display percentile')
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/'qa').mkdir(exist_ok=True)
    summary=json.loads((args.evidence/'summary.json').read_text())
    scoped=False
    if args.constrained_scope:
        acceptance=json.loads((args.evidence/'scoped_acceptance.json').read_text())
        required=['coordinate_gradient_passed','physical_quality_screen_passed',
                  'parameterization_fixed_before_comparison']
        if not all(acceptance.get(key) is True for key in required):
            raise ValueError('Independent constrained-coordinate/quality acceptance is incomplete')
        if acceptance.get('original_all_physical_gate_passed') is not False or summary.get('passed') is not False:
            raise ValueError('Use full acceptance for an all-physical pass; never rewrite the original failed gate')
        if acceptance.get('summary_sha256')!=sha(args.evidence/'summary.json') or acceptance.get('raw_kernels_sha256')!=sha(args.evidence/'raw_kernels.npz'):
            raise ValueError('Scoped acceptance does not bind the current numerical evidence')
        if not 0<=acceptance.get('max_coordinate_relative_l2',float('inf'))<=summary['config']['tolerance']:
            raise ValueError('Coordinate gradients exceed the original numerical tolerance')
        scoped=True
    elif not summary.get('passed'):
        raise ValueError('Only plot a checked AD/Hybrid pair as a verified demonstration')
    quality_file=args.evidence/'physical_quality.json'
    if quality_file.exists():
        quality=json.loads(quality_file.read_text())
        if quality.get('usable_for_geological_detail') is False:
            raise ValueError('Numerical AD agreement does not override a rejected forward-quality case')
    cfg=summary['config']
    source=np.load(args.evidence/'raw_kernels.npz',allow_pickle=False)
    model=np.load(args.model,allow_pickle=False)
    x,z=source['x_m']/1000,source['z_m']/1000
    if not np.array_equal(model['x']/1000,x) or not np.array_equal(model['z']/1000,z):
        raise ValueError('Model and actual gradient coordinates differ')
    mask=source['parameter_mask' if 'parameter_mask' in source else 'active_mask']
    active=mask>0;core=mask>.999
    def gradient(method,name,constrained=False):
        canonical=f'gradient_{method}_{"coordinate_" if constrained else ""}{name}'
        legacy=f'{method}_{"active_" if constrained else ""}gradient_{name}'
        return source[canonical if canonical in source else legacy]
    if mask.shape!=(len(z),len(x)) or not core.any():
        raise ValueError('Invalid declared active parameterization')
    contract=dict(claim=('AD and Hybrid agree for the predeclared constrained inversion coordinates; the original full-physical gradient gate failed and remains failed.' if scoped else 'A refined-grid marine-pressure experiment exposes spatial objective derivatives while AD and Hybrid agree in both physical and constrained coordinates.'),
        backend='python',archetype='image plate + quant',
        comparison='Different grid/acquisition/starting model from the old performance benchmark; no performance improvement score.',
        panels=['true input','actual Gaussian initial input','AD active-coordinate derivative','Hybrid active-coordinate derivative'],
        integrity='No derivative smoothing, illumination division, sharpening or inserted model texture. Only explicitly manifested display saturation.',
        export='Editable PDF/SVG and 300 dpi PNG; >5pt text; measured alignment and rendered collision QA.')
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    records={}
    # Keep the physical aspect ratio: the ROI spans all benchmark depths.
    fig,axes=plt.subplots(2,4,figsize=(11.8,10.1));bars=[]
    for row,name in enumerate(['vp','vs']):
        display={'vp':'Vp','vs':'Vs'}[name]
        truth,initial=model[name+'_true']/1000,model[name+'_initial']/1000
        if truth.shape!=mask.shape:
            raise ValueError('Model shape differs from the actual derivative grid')
        lo,hi=float(min(truth.min(),initial.min())),float(max(truth.max(),initial.max()))
        for col,array in enumerate([truth,initial]):
            image=axes[row,col].pcolormesh(x,z,array,shading='nearest',cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
            decorate(axes[row,col],x,z)
            axes[row,col].set_title(f'{"abcdefgh"[row*4+col]}  {display}: {["True","Initial"][col]}',loc='left',pad=9)
            bars.append((image,axes[row,col],f'{display} (km/s)',[lo,(lo+hi)/2,hi]))
        gradients=[gradient(method,name,constrained=True) for method in ['autograd','hybrid']]
        limit=float(np.percentile(np.abs(np.concatenate([g[core] for g in gradients])),args.percentile))
        if not np.isfinite(limit) or limit<=0:
            raise ValueError('Empty gradient scale')
        for col,(method,g) in enumerate(zip(['Native AD','Hybrid'],gradients),start=2):
            image=axes[row,col].pcolormesh(x,z,np.ma.masked_where(~active,g),shading='nearest',
                cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            axes[row,col].set_facecolor('#e6e9ed')
            decorate(axes[row,col],x,z)
            axes[row,col].set_title(f'{"abcdefgh"[row*4+col]}  {method}: {display}',loc='left',pad=9)
            bars.append((image,axes[row,col],'dJ/dθ (objective / [m/s])',[-limit,0,limit]))
        records[name]=dict(display='linear symmetric percentile over full-weight coordinates; same scale for AD and Hybrid',
            percentile=args.percentile,absolute_limit=limit,gradient_values_modified=False,
            saturated_fraction_of_active=[float(np.mean(abs(g[active])>limit)) for g in gradients],
            relative_l2=float(np.linalg.norm(gradients[1]-gradients[0])/np.linalg.norm(gradients[0])))
    shots=len(cfg['selected_shots'])
    scope_line='Verified declared-coordinate derivatives; full-physical 1e−8 gate NOT met' if scoped else f'{shots} independently evaluated shot(s)'
    fig.suptitle(f'Marmousi-II marine-pressure gradients · {scope_line}\n'
                 f'{cfg["f0"]:g} Hz · {cfg["grid"]["dx"]:g} m grid · σ = {cfg["initial_smoothing_sigma_m"]:g} m · {shots} shot(s) · fixed water/buffer',
                 y=.99,fontsize=11)
    fig.subplots_adjust(left=.065,right=.975,top=.91,bottom=.115,wspace=.46,hspace=.53)
    export(fig,axes,output,'models_and_active_gradients',bars)

    # Show both coordinate meanings explicitly, including the constant-density
    # control. A mask is a static model map here, not an image-cleaning trick.
    fig,axes=plt.subplots(2,3,figsize=(10.3,9.8));bars=[]
    for col,name in enumerate(['vp','vs','rho']):
        display={'vp':'Vp','vs':'Vs','rho':'Density'}[name]
        for row,meaning in enumerate(['Full physical g','Actual Pᵀg']):
            g=gradient('hybrid',name,constrained=(row==1))
            limit=float(np.percentile(np.abs(g[core]),args.percentile))
            if limit<=0:
                raise ValueError('All-zero derivative')
            image=axes[row,col].pcolormesh(x,z,g if row==0 else np.ma.masked_where(~active,g),
                shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            axes[row,col].set_facecolor('#e6e9ed');decorate(axes[row,col],x,z)
            axes[row,col].set_title(f'{"abcdef"[row*3+col]}  {display}: {meaning}',loc='left',pad=9)
            unit='objective / [kg/m³]' if name=='rho' else 'objective / [m/s]'
            bars.append((image,axes[row,col],unit,[-limit,0,limit]))
            selected=np.ones_like(active,dtype=bool) if row==0 else active
            records[f'{name}_{"physical" if row==0 else "constrained"}']=dict(
                display='linear symmetric percentile over full-weight coordinates',
                percentile=args.percentile,absolute_limit=limit,gradient_values_modified=False,
                saturated_fraction_of_displayed=float(np.mean(abs(g[selected])>limit)))
    physical_title='Diagnostic full physical g (strict gate failed) and verified declared-coordinate Pᵀg' if scoped else 'Physical derivatives and declared active-coordinate derivatives'
    fig.suptitle(physical_title+'\n'
                 'Gray = fixed coordinates; density truth is constant, so its derivative is not a density image',y=.99,fontsize=10)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.115,wspace=.43,hspace=.53)
    export(fig,axes,output,'physical_and_active_gradients',bars)
    manifest=dict(status='rendered; independent PDF/visual QA required',model_sha256=sha(args.model),
        gradient_bundle_sha256=sha(args.evidence/'raw_kernels.npz'),summary_sha256=sha(args.evidence/'summary.json'),
        plotter_sha256=sha(__file__),config=cfg,gradient_scales=records,
        coordinates='physical arrays and diagonal static-map P^T derivatives preserved separately',
        acceptance_scope='declared constrained-coordinate gradients only; original full-physical numerical gate failed' if scoped else 'full physical and constrained-coordinate gradients',
        scoped_acceptance_sha256=sha(args.evidence/'scoped_acceptance.json') if scoped else None,
        fixed_fraction=float(np.mean(~active)),fully_active_fraction=float(np.mean(core)),
        source_inputs_distributed=False,gradient_postprocessing='none; only declared image saturation',
        numerical_accuracy_scope='Uniform-medium FD8 dispersion analysis and exact discrete AD/Hybrid comparison; not a continuum or interface-convergence study',
        model_display='local source inputs, no model texture inserted into derivative panels')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
