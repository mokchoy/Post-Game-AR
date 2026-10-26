import json, sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step48_analyze_lib import *
R={r['i']:r for r in json.load(open(O+'res.json'))}
frames=[996,660,1203,1452,1515,2019,2397]
tiles=[]
def lab(im,t): cv2.putText(im,t,(4,18),0,.5,(255,255,255),2); cv2.putText(im,t,(4,18),0,.5,(0,0,0),1); return im
for i in frames:
    r=R[i]; T=np.array(rows[i]['render_T_CW']); d=D[str(i)].astype(float); bgr=cv2.imread(O+f'img/{i:05d}.jpg')
    zp,Xp=plane_hit(T,np.array([0,1.,0]),np.array([0,EL,0])); pan=np.isfinite(zp)&(np.abs(Xp[...,0])<=W/2)&(Xp[...,2]>=0)&(Xp[...,2]<=H)
    hand=skin(bgr); margin=np.maximum(.1,.1*np.nan_to_num(zp))
    a=bgr.copy(); cv2.drawContours(a,cv2.findContours(hand.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
    cv2.drawContours(a,cv2.findContours(pan.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),2)
    dn=cv2.applyColorMap(np.clip((d-d.min())/(np.ptp(d)+1e-6)*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    row=[lab(a,f"{i} skin=yellow panel=green"),lab(dn,f"model d {d.min():.1f}..{d.max():.1f}")]
    for name,t,s in [('logged',r['t_log'],r['s_log']),('table',r['t_tab'],r['s_tab'])]:
        with np.errstate(divide='ignore',invalid='ignore'): Z=np.where(d-t>1e-3,s/(d-t),np.inf)
        zc=cv2.applyColorMap((np.clip((np.log(np.clip(Z,.3,5))-np.log(.3))/np.log(5/.3),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        hid=pan&(Z+margin<zp); zc[hid]=(zc[hid]*.3+np.array([255,255,255])*.7).astype(np.uint8)
        cv2.drawContours(zc,cv2.findContours(pan.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,255,255),1)
        row.append(lab(zc,f"Z {name} t={t:.1f} s={s:.1f} hid={hid[pan].mean() if pan.any() else 0:.2f}"))
    tiles.append(np.hstack([cv2.resize(x,(320,240)) for x in row]))
cv2.imwrite(O+'sheet2.jpg',np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,85])
