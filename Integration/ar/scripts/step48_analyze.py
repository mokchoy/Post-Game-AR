"""Step 48 pre-check analysis: per-frame table-plane fit vs logged fit, hand proxy = skin colour (approximate). Needs step48_infer.py output."""
import json, sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step48_analyze_lib import *
res=[]
for i in idx:
    r=rows[i]; o=r['diagnostics']['occlusion']; T=np.array(r['render_T_CW'])
    d=D[str(i)].astype(np.float64); bgr=cv2.imread(O+f'img/{i:05d}.jpg')
    # panel: plane Y=EL (world), vertical; rectangle X in [-W/2,W/2], Z in [0,H]
    zp,Xp=plane_hit(T,np.array([0,1.,0]),np.array([0,EL,0]))
    pan=np.isfinite(zp)&(np.abs(Xp[...,0])<=W/2)&(Xp[...,2]>=0)&(Xp[...,2]<=H)
    zt,Xt=plane_hit(T,np.array([0,0,1.]),np.zeros(3))
    hand=skin(bgr); handd=cv2.dilate(hand.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
    tab=np.isfinite(zt)&(zt<5)&(np.abs(Xt[...,0])<1.0)&(Xt[...,1]>-0.6)&(Xt[...,1]<0.9)&~handd
    tab[::1,1::2]=False; tab[1::2,:]=False
    t0,s0=o['fit']; margin=np.maximum(.1,.1*zp)
    with np.errstate(divide='ignore',invalid='ignore'):
        Zlog=np.where(d-t0>1e-3,s0/(d-t0),np.inf)
    hid_log=pan&(Zlog+margin<zp)
    rec={'i':i,'t_log':t0,'s_log':s0,'src':o['alignment_source'],'hid_logged':o.get('hidden_fraction'),
         'pan_px':int(pan.sum()),'hand_px':int(hand.sum()),'hand_pan_px':int((hand&pan).sum()),
         'hid_log_rep':float(hid_log[pan].mean()) if pan.any() else np.nan}
    if tab.sum()>300:
        iz=1/zt[tab]; p,inl,sc=robust_fit(iz,d[tab]); tt,ss=p
        izi=iz[inl]; rec.update(t_tab=tt,s_tab=ss,tab_n=int(inl.sum()),tab_inl=float(inl.mean()),tab_span=float((np.percentile(izi,95)-np.percentile(izi,5))/np.median(izi)),
             tab_res_rel=float(sc/ss), tab_zmin=float(1/np.percentile(izi,95)),tab_zmax=float(1/np.percentile(izi,5)))
        with np.errstate(divide='ignore',invalid='ignore'):
            Ztab=np.where(d-tt>1e-3,ss/(d-tt),np.inf)
        hid_tab=pan&(Ztab+margin<zp)
        rec['hid_tab']=float(hid_tab[pan].mean()) if pan.any() else np.nan
        hp=hand&pan; nh=pan&~handd
        if hp.sum()>200:
            rec.update(hand_z_log=float(np.median(Zlog[hp])),hand_z_tab=float(np.median(Ztab[hp])),
                       hand_hid_log=float(hid_log[hp].mean()),hand_hid_tab=float(hid_tab[hp].mean()),hand_d=float(np.median(d[hp])))
        if nh.sum()>500:
            rec.update(bg_falsehide_log=float(hid_log[nh].mean()),bg_falsehide_tab=float(hid_tab[nh].mean()))
        # far background: upper image, not table, not hand, above horizon-ish -> beyond table region
        far=(~np.isfinite(zt)|(zt>3.0))&~handd; far[300:]=False
        if far.sum()>2000:
            rec['far_d']=float(np.median(d[far])); rec['far_z_tab']=float(np.median(Ztab[far])); rec['far_z_log']=float(np.median(Zlog[far]))
            rec['far_d_rel']=float((np.median(d[far])-tt)/ss)  # = 1/Z under table fit
        rec['d_p99']=float(np.percentile(d,99)); rec['d_tab_med']=float(np.median(d[tab]))
    res.append(rec)
json.dump(res,open(O+'res.json','w'))
print(len(res))
