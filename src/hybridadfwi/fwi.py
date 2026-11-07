"""Data-only objectives and bounded optimizer utilities for serial-shot FWI.

No wave-equation kernels live here. Optimizer functions never receive true Vp.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def lowpass(data, dt, cutoff_hz=None, transition_hz=.5):
    """Fixed differentiable zero-phase FFT low-pass, with one record of zeros per end.

    Raised-cosine response is unity below cutoff-transition and zero above
    cutoff+transition. Padding prevents circular end-to-start wrap in the record.
    """
    if cutoff_hz is None:
        return data
    if cutoff_hz <= transition_hz or transition_hz <= 0:
        raise ValueError("Require cutoff > positive transition")
    nt=data.shape[1]
    extended=F.pad(data.transpose(1,2),(nt,nt))
    frequency=torch.fft.rfftfreq(3*nt,dt,device=data.device,dtype=data.dtype)
    u=((frequency-(cutoff_hz-transition_hz))/(2*transition_hz)).clamp(0,1)
    response=.5+.5*torch.cos(torch.pi*u)
    filtered=torch.fft.irfft(torch.fft.rfft(extended,dim=-1)*response,n=3*nt,dim=-1)
    return filtered[...,nt:2*nt].transpose(1,2)


def data_loss(predicted, observed, window, scale, dt, cutoff_hz=None, transition_hz=.5):
    return per_shot_losses(predicted,observed,window,scale,dt,cutoff_hz,transition_hz).mean()


def per_shot_losses(predicted, observed, window, scale, dt, cutoff_hz=None, transition_hz=.5):
    """Each loss divides by Nt*Nr once; callers choose shot sum versus mean."""
    residual=lowpass(predicted-observed,dt,cutoff_hz,transition_hz)
    return .5*((window*residual)/scale).square().mean(dim=(1,2))


def native_no_temporal_graph(prop,shot_ids,checkpoint_segments=84):
    """Native loss-only forward with the SAME detached-step execution as Hybrid.

    Static material preparation keeps its normal grad mode, avoiding a distinct
    coefficient-preparation execution path. Native step kernels execute no_grad;
    no temporal checkpoints or adjoint tape are retained. Vendor source intact.
    """
    from .adjoint import _kernels,_HOOK_LOCK,preserve_padding_dtype
    kernels=_kernels()
    def execute(function,*args,**kwargs):
        detached=tuple(value.detach() if torch.is_tensor(value) else value for value in args)
        with torch.no_grad():
            return function(*detached)
    with _HOOK_LOCK,preserve_padding_dtype():
        original=kernels.checkpoint;kernels.checkpoint=execute
        try:
            with torch.enable_grad():
                output=prop.forward(fd_order=8,checkpoint_segments=checkpoint_segments,shot_index=list(shot_ids))
        finally:
            kernels.checkpoint=original
    if any(value.requires_grad for value in output.values() if torch.is_tensor(value)):
        raise RuntimeError("A temporal graph escaped the loss-only forward")
    return output


def lbfgs_direction(gradient, history, initial_scale=1.):
    """Two-loop L-BFGS; history consists only of accepted (s,y) vectors."""
    q=np.array(gradient,copy=True)
    alphas=[]
    for s,y in reversed(history):
        rho=1./float(np.vdot(s,y))
        alpha=rho*float(np.vdot(s,q));alphas.append(alpha);q-=alpha*y
    if history:
        s,y=history[-1]
        gamma=float(np.vdot(s,y)/np.vdot(y,y))
    else:
        gamma=float(initial_scale)
    r=gamma*q
    for (s,y),alpha in zip(history,reversed(alphas)):
        rho=1./float(np.vdot(s,y))
        r+=s*(alpha-rho*float(np.vdot(y,r)))
    direction=-r
    if not np.isfinite(direction).all() or float(np.vdot(direction,gradient))>=0:
        return -np.array(gradient,copy=True),False
    return direction,True


def bounds(reference_vp, fixed_vs, mask, velocity_scale, upper_vp=6000., bulk_speed_floor=100.):
    """Coordinate bounds enforcing positive 3-D bulk with known Vs only."""
    lower=np.sqrt(4./3*fixed_vs**2+bulk_speed_floor**2)
    active=mask>0
    if np.any(reference_vp[~active]<lower[~active]) or np.any(reference_vp[~active]>upper_vp):
        raise ValueError("Fixed reference cells violate declared physical bounds")
    lo=np.zeros_like(mask);hi=np.zeros_like(mask)
    lo[active]=(lower[active]-reference_vp[active])/(velocity_scale*mask[active])
    hi[active]=(upper_vp-reference_vp[active])/(velocity_scale*mask[active])
    return lo,hi


def project(q, lo, hi):
    return np.minimum(np.maximum(q,lo),hi)


def physical_vp(reference,mask,q,velocity_scale):
    coordinate=reference+velocity_scale*q
    return reference+mask*(coordinate-reference)


def candidate(q,direction,alpha,lo,hi,reference,mask,velocity_scale,maximum_update):
    """Project first, then cap the actual mapped physical change from anchor."""
    trial=project(q+alpha*direction,lo,hi)
    anchor=physical_vp(reference,mask,q,velocity_scale)
    peak=float(np.max(np.abs(physical_vp(reference,mask,trial,velocity_scale)-anchor)))
    if peak>maximum_update:
        trial=q+(trial-q)*(maximum_update/peak)
    actual=physical_vp(reference,mask,trial,velocity_scale)-anchor
    if np.max(np.abs(actual))>maximum_update*(1+1e-10):raise AssertionError("Physical update cap failed")
    return trial,actual


def cap_direction(direction, mask, velocity_scale, maximum_physical_update):
    direction=np.array(direction,copy=True)
    direction[mask==0]=0.
    peak=float(np.max(np.abs(mask*velocity_scale*direction)))
    if peak>maximum_physical_update:
        direction*=maximum_physical_update/peak
    return direction


def accepted_curvature(s,y,relative_floor=1e-10):
    sy=float(np.vdot(s,y));scale=float(np.linalg.norm(s)*np.linalg.norm(y))
    return np.isfinite(sy) and sy>relative_floor*scale and sy>0
