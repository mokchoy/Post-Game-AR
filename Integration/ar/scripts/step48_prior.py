import json, sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step48_analyze_lib import *
idx=sorted(int(k) for k in D.files)
S=4; h,w=480//S,640//S
Ks=K.copy(); Ks[:2]/=S; Ks[:2,2]=(K[:2,2]+.5)/S-.5
ys,xs=np.indices((h,w)); rs=np.stack((xs,ys,np.ones_like(xs)),-1).reshape(-1,3)@np.linalg.inv(Ks).T
def warp(P,C,T0,T1):
    ok=np.isfinite(P).ravel(); X=rs[ok]*P.ravel()[ok,None]
    Xw=(X-T0[:3,3])@T0[:3,:3]; Xc=Xw@T1[:3,:3].T+T1[:3,3]; z=Xc[:,2]; good=z>.1
    uv=Xc[good]@Ks.T; u=np.rint(uv[:,0]/uv[:,2]).astype(int); v=np.rint(uv[:,1]/uv[:,2]).astype(int); z=z[good]; c=C.ravel()[ok][good]
    inb=(u>=0)&(u<w)&(v>=0)&(v<h); u,v,z,c=u[inb],v[inb],z[inb],c[inb]
    o=np.argsort(-z); Pn=np.full((h,w),np.nan); Cn=np.zeros((h,w))
    Pn[v[o],u[o]]=z[o]; Cn[v[o],u[o]]=c[o]          # nearest wins (written last)
    # fill 1-px splat holes
    m=np.isnan(Pn); f=cv2.dilate(np.nan_to_num(Pn,nan=0).astype(np.float32),np.ones((3,3),np.uint8))  # crude
    Pn[m&(f>0)]=np.nan
    return Pn,Cn
def fit(iz,d):
    A=np.stack([np.ones_like(iz),iz],1); wt=np.ones_like(d)
    for _ in range(15):
        p=np.linalg.lstsq(A*wt[:,None],d*wt,rcond=None)[0]; r=d-A@p
        sc=1.4826*np.median(np.abs(r))+1e-6; wt=np.sqrt(np.minimum(1,1.5*sc/np.maximum(np.abs(r),1e-9)))
    return p
R0={r['i']:r for r in json.load(open(O+'res.json'))}
P=None; out=[]
for i in idx:
    r=rows[i]; T=np.array(r['render_T_CW']); d=D[str(i)].astype(float); bgr=cv2.imread(O+f'img/{i:05d}.jpg')
    ds=cv2.resize(d,(w,h),interpolation=cv2.INTER_AREA)
    zt,Xt=plane_hit(T,np.array([0,0,1.]),np.zeros(3)); zts=zt[S//2::S,S//2::S]; Xts=Xt[S//2::S,S//2::S]
    table=np.isfinite(zts)&(zts<5)&(np.abs(Xts[...,0])<1.0)&(Xts[...,1]>-0.6)&(Xts[...,1]<0.9)
    if P is not None: Pw,Cw=warp(P,C,Tp,T)
    else: Pw,Cw=np.full((h,w),np.nan),np.zeros((h,w))
    ref=np.where(np.isfinite(Pw)&(Cw>=3),Pw,np.where(table,zts,np.nan)); ok=np.isfinite(ref)
    t,s=fit(1/ref[ok],ds[ok])
    with np.errstate(divide='ignore',invalid='ignore'):
        Z=np.where(d-t>1e-3,s/(d-t),np.inf); Zs=np.where(ds-t>1e-3,s/(ds-t),np.inf)
    # occluder test
    zp,Xp=plane_hit(T,np.array([0,1.,0]),np.array([0,EL,0])); pan=np.isfinite(zp)&(np.abs(Xp[...,0])<=W/2)&(Xp[...,2]>=0)&(Xp[...,2]<=H)
    margin=np.maximum(.1,.1*np.nan_to_num(zp))
    refu=cv2.resize(np.nan_to_num(ref,nan=0).astype(np.float32),(640,480),interpolation=cv2.INTER_NEAREST)
    near_bg=(refu>0)&(Z<0.8*refu)
    hid_a=pan&(Z+margin<zp)                     # fit vs prior only
    hid_b=hid_a&((refu==0)|near_bg)             # plus gate: must be nearer than expected background
    hid_c=pan&near_bg&(Z<zp)
    hid_d=pan&cv2.morphologyEx(near_bg.astype(np.uint8),cv2.MORPH_OPEN,np.ones((7,7),np.uint8)).astype(bool)&(Z<zp+.1)
    # update prior
    nearblob=cv2.dilate(((np.isfinite(ref))&(Zs<0.8*np.nan_to_num(ref,nan=99))|(Zs<0.6)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    Pn=Pw.copy(); Cn=Cw.copy()
    agree=np.isfinite(Pw)&(np.abs(Zs-Pw)<.15*Pw)
    Pn[agree]=(Pw[agree]*Cn[agree]+Zs[agree])/(Cn[agree]+1); Cn[agree]=np.minimum(Cn[agree]+1,20)
    new=~np.isfinite(Pw)&np.isfinite(Zs)&(Zs<8)&~nearblob
    Pn[new]=Zs[new]; Cn[new]=1
    far=np.isfinite(Pw)&(Zs>1.15*Pw); Cn[far]-=1; Pn[Cn<=0]=np.nan
    P,C,Tp=Pn,Cn,T
    hand=skin(bgr); handd=cv2.dilate(hand.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool); hp=hand&pan; nh=pan&~handd
    rec={'i':i,'t':t,'s':s,'prior_cov':float(np.mean(np.isfinite(Pw)&(Cw>=3)))}
    if hp.sum()>1500: rec.update(hand_c=float(hid_c[hp].mean()),hand_d=float(hid_d[hp].mean()),hand_a=float(hid_a[hp].mean()),hand_b=float(hid_b[hp].mean()),hand_z=float(np.median(Z[hp])))
    if nh.sum()>500: rec.update(bg_c=float(hid_c[nh].mean()),bg_d=float(hid_d[nh].mean()),bg_a=float(hid_a[nh].mean()),bg_b=float(hid_b[nh].mean()))
    out.append(rec)
json.dump(out,open(O+"prior2.json","w"))
A={r['i']:r for r in out}
H=[i for i in A if 'hand_a' in A[i] and 'hand_hid_log' in R0[i]]
B=[i for i in A if 'bg_a' in A[i] and 'bg_falsehide_log' in R0[i]]
g=lambda L,f:np.array([f(i) for i in L])
print("prior coverage median",np.median([r['prior_cov'] for r in out]))
print("hand-on-panel frames",len(H))
for name,f in [('logged',lambda i:R0[i]['hand_hid_log']),('table',lambda i:R0[i]['hand_hid_tab']),('prior fit',lambda i:A[i]['hand_a']),('prior fit+gate',lambda i:A[i]['hand_b']),('fg blob, Z<panel',lambda i:A[i]['hand_c']),('fg blob, Z<panel+.1',lambda i:A[i]['hand_d'])]:
    v=g(H,f); print("%-15s hand hidden mean %.2f  frames <50%% hidden %.2f"%(name,v.mean(),np.mean(v<.5)))
for name,f in [('logged',lambda i:R0[i]['bg_falsehide_log']),('table',lambda i:R0[i]['bg_falsehide_tab']),('prior fit',lambda i:A[i]['bg_a']),('prior fit+gate',lambda i:A[i]['bg_b']),('fg blob, Z<panel',lambda i:A[i]['bg_c']),('fg blob, Z<panel+.1',lambda i:A[i]['bg_d'])]:
    v=g(B,f); print("%-15s bg false hide mean %.4f  frames >2%% %.2f"%(name,v.mean(),np.mean(v>.02)))
print("hand Z median prior fit",np.median(g(H,lambda i:A[i]['hand_z'])),"table",np.median(g(H,lambda i:R0[i]['hand_z_tab'])),"logged",np.median(g(H,lambda i:R0[i]['hand_z_log'])))
