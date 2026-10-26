import json, numpy as np, cv2
import os
US=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(US)
U=ROOT+'/runs/20261025T092056-7a6e0212/'
O=ROOT+'/runs/step48-092056/'   # depth.npz and img/ live here (step48_infer.py)
rows=[json.loads(l) for l in open(U+'frames.jsonl')]
D=np.load(O+'depth.npz'); idx=sorted(int(k) for k in D.files)
K=np.array([[597.1328577780656,0,317.1013147461914],[0,599.8120170977288,226.35911097047054],[0,0,1]])
yy,xx=np.indices((480,640)); rays=np.stack((xx,yy,np.ones_like(xx)),-1)@np.linalg.inv(K).T
W,H,EL=1.0,.5625,.15
def plane_hit(T,n_w,p_w):
    R,t=T[:3,:3],T[:3,3]; n=R@n_w; p=R@p_w+t
    with np.errstate(divide='ignore',invalid='ignore'): Z=(n@p)/(rays@n)
    Z[~np.isfinite(Z)|(Z<=0)]=np.nan
    Xw=((rays*Z[...,None])-t)@R   # world coords
    return Z,Xw
def skin(bgr):
    ycc=cv2.cvtColor(bgr,cv2.COLOR_BGR2YCrCb); Y,Cr,Cb=[ycc[...,i].astype(int) for i in range(3)]
    m=((Cr>=140)&(Cr<=180)&(Cb>=80)&(Cb<=125)&(Y>40)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    keep=np.zeros_like(m,bool)
    for j in range(1,n):
        if st[j,4]>2500: keep|=lab==j
    return keep
def robust_fit(iz,d):
    A=np.stack([np.ones_like(iz),iz],1); w=np.ones_like(d)
    for _ in range(15):
        p=np.linalg.lstsq(A*w[:,None],d*w,rcond=None)[0]; r=d-A@p
        sc=1.4826*np.median(np.abs(r))+1e-6; w=np.sqrt(np.minimum(1,1.5*sc/np.maximum(np.abs(r),1e-9)))
    inl=np.abs(r)<2.5*sc
    return p,inl,sc
