"""Small fresh P-SV Vp-only pilot; no Marmousi input or GPU required."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import gradient_detail as detail
import gradient_detail_vponly as vp_only
from hybridadfwi.adjoint import hybrid_forward


def test_fixed_true_vs_rho_and_fresh_vp_adjoint(tmp_path):
    torch.set_num_threads(1)
    nz,nx=24,32
    z,x=np.indices((nz,nx),dtype=float)
    water=z<5
    truth={"vp":np.where(water,1500.,2450.+12*z+2*x),
           "vs":np.where(water,0.,1250.+4*z+x),"rho":np.full((nz,nx),2250.)}
    arrays={}
    for name,value in truth.items():
        arrays[name+"_true"]=value
        arrays[name+"_initial"]=np.where(water,value,.97*value) if name=="vp" else value.copy()
    path=tmp_path/"vp-only.npz"
    np.savez(path,**arrays,dx=6.25,dz=6.25,ox=0.,oz=0.)
    cfg=dict(model=str(path),device="cpu",nt=320,dt=.0005,f0=40.,shots=3,
             selected_shots=[1],receivers=8,depth_m=12.5,nabc=8,free_surface=False,
             segments=8,parameters=["vp"],parameter_mode="vp_only_true_vs_true_rho",
             fixed_pml_reference_vmax_m_s=4700.,constrained_coordinates=True,
             fixed_buffer_m=12.5,parameter_taper_m=12.5,minimum_offset_m=0.,
             offset_taper_m=6.25,mute_velocity_m_s=1500.,direct_mute_after_s=0.,
             time_taper_s=.001,end_taper_s=.005)
    assert vp_only.audit_model(cfg)["checks"]["only_vp_trainable"]
    with torch.no_grad():
        _,prop,_,_=detail.build_model(cfg,truth=True,trainable=False)
        obs=detail.pressure(detail.native(prop,cfg,1))
        window=detail.data_window(prop,cfg,1,torch)
        scale=(obs*window).square().mean().sqrt().item()
    model,prop,coords,mask=detail.build_model(cfg)
    out=detail.native(prop,cfg,1)
    loss=detail.loss_function(detail.pressure(out),obs,window,scale)
    loss.backward()
    assert not model.vs.requires_grad and not model.rho.requires_grad
    assert model.vs.grad is None and model.rho.grad is None
    np.testing.assert_array_equal(model.vs.detach().numpy(),truth["vs"])
    np.testing.assert_array_equal(model.rho.detach().numpy(),truth["rho"])
    physical=model.vp.grad.clone()
    coordinate=coords["vp"].grad.clone()
    torch.testing.assert_close(coordinate,mask*physical,rtol=0,atol=0)
    other,other_prop,_,_=detail.build_model(cfg)
    ho,tape=hybrid_forward(other_prop,fd_order=8,checkpoint_segments=8,shot_index=[1])
    txx,tzz=(ho[k].detach().requires_grad_(True) for k in ("txx","tzz"))
    hp=-.5*(txx+tzz)
    hloss=detail.loss_function(hp,obs,window,scale)
    sx,sz=torch.autograd.grad(hloss,(txx,tzz))
    hg=tape.backward({"txx":sx,"tzz":sz})
    assert set(hg)=={"vp"}
    error=(hg["vp"]-physical).norm()/physical.norm()
    assert error.item()<1e-8
    torch.testing.assert_close(hp,detail.pressure(out),rtol=0,atol=0)
    initial=coords["vp"].detach().clone()
    direction=coordinate/coordinate.norm()
    expected=(coordinate*direction).sum().item()
    values=[]
    for sign in (-1,1):
        with torch.no_grad():
            coords["vp"].copy_(initial+sign*.05*direction)
            p=detail.pressure(detail.native(prop,cfg,1))
            values.append(detail.loss_function(p,obs,window,scale).item())
            np.testing.assert_array_equal(model.vs.numpy(),truth["vs"])
            np.testing.assert_array_equal(model.rho.numpy(),truth["rho"])
    fd=(values[1]-values[0])/.1
    assert abs(fd-expected)/abs(expected)<1e-5
