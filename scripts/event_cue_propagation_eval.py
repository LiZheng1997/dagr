"""Between-frame cue update from events alone, on the DSEC night sequences of Table V.

Anchor: the frame-only DAGR box of each labelled track at image time t0 (lt2 *_evals.jsonl,
score 0.3, matched to the label). Between t0 and t0+50 ms three cues are compared against the
label interpolated to t0+d (d = 10..50 ms):
  held     - the anchor box unchanged (what a frame-rate detector shows between frames)
  dagr     - the event-and-frame DAGR box recomputed with events up to t0+d (offline, from lt2)
  prop     - the anchor box moved by events only: every STEP ms the median position of the events
             inside the (margin-expanded) current box over the last WIN ms is compared with the
             previous median and the box is translated by the difference
  propsc   - prop plus a scale update from the ratio of the event spread (MAD) in x and y
Also: a class-agnostic activity indicator (event count in the left vs the right half of the
frame per window) scored against the side of the nearest labelled pedestrian.
Per-update compute time of prop and of the activity indicator is measured on this CPU.
Output: /output/evcue/evcue_<seq>.jsonl (one record per track instant) and a printed summary."""
import sys, os, json, time
sys.argv=["x","--config","config/dagr-s-dsec.yaml","--use_image","--img_net","resnet50","--checkpoint","data/dagr_s_50.pth","--batch_size","1","--n_nodes","20000","--dataset_directory","/dsec_night","--output_directory","/tmp/x"]
sys.path.insert(0,"/opt/dagr/src")
import numpy as np
from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
SEQS=os.environ.get("EV_SEQS","zurich_city_09_b,zurich_city_09_c,zurich_city_09_d,zurich_city_10_b").split(",")
STEP=int(os.environ.get("EV_STEP_US","5000")); WIN=int(os.environ.get("EV_WIN_US","5000"))
STEPS=[10000,20000,30000,40000,50000]; THR="0.3"; MINEV=int(os.environ.get("EV_MIN","12"))
MAXP=int(os.environ.get("EV_MAX_PAIRS","0"))
NOISE_HZ=float(os.environ.get("EV_NOISE_HZ","0")); RNG=np.random.default_rng(0); TAG=os.environ.get("EV_TAG","")
CLS={0:"pedestrian",2:"car"}
args=FLAGS()
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return i/u if u>0 else 0.0
def cerr(a,b): return float(np.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2))
def inbox(x,y,b,m):
    return (x>=b[0]-m)&(x<=b[2]+m)&(y>=b[1]-m)&(y<=b[3]+m)

XWIN=int(os.environ.get("EV_XWIN_US","5000")); SEARCH=int(os.environ.get("EV_SEARCH","6")); BLUR=float(os.environ.get("EV_BLUR","1.0"))
import cv2
def hist(xx,yy,x0,y0,W,H):
    h=np.zeros((H,W),np.float32)
    xi=(xx-x0).astype(np.int32); yi=(yy-y0).astype(np.int32); k=(xi>=0)&(xi<W)&(yi>=0)&(yi<H)
    h+=np.bincount(yi[k]*W+xi[k],minlength=H*W).reshape(H,W).astype(np.float32)
    return cv2.GaussianBlur(h,(0,0),BLUR) if BLUR>0 else h
def xc_shift(ref,cur):
    """normalised cross-correlation (cv2.matchTemplate) over +-SEARCH px, parabolic sub-pixel"""
    H,W=ref.shape; S=SEARCH
    tpl=ref[S:H-S,S:W-S]
    if tpl.shape[0]<2 or tpl.shape[1]<2: return 0.0,0.0,0.0,0.0
    sc=cv2.matchTemplate(cur,tpl,cv2.TM_CCORR_NORMED)
    iy,ix=np.unravel_index(int(np.argmax(sc)),sc.shape)
    def sub(a,b,c):
        d=a-2*b+c
        return 0.0 if abs(d)<1e-9 else 0.5*(a-c)/d
    fx=sub(sc[iy,ix-1],sc[iy,ix],sc[iy,ix+1]) if 0<ix<sc.shape[1]-1 else 0.0
    fy=sub(sc[iy-1,ix],sc[iy,ix],sc[iy+1,ix]) if 0<iy<sc.shape[0]-1 else 0.0
    return ix-S+fx, iy-S+fy, float(sc[iy,ix]), float(sc[S,S])

out_all=[]; txc=[]; tprop=[]; tact=[]; act_hits=[0,0]
for seq in SEQS:
    recs={}
    for l in open("/output/lt2_%s_evals.jsonl"%seq):
        r=json.loads(l); recs.setdefault(r["pair"],{})[r["model"] if r["model"]=="frame" else r["t"]-r["t0"]]=r
    S=json.load(open("/output/lt2_%s_summary.json"%seq)); cls_of={int(r["track"]):r["cls"] for r in S["by_score"][THR]["tracks"]}
    ds=DSEC(__import__("pathlib").Path("/dsec_night/one_"+seq),"test",Augmentations.transform_testing,debug=False,min_bbox_diag=15,min_bbox_height=10,demo=True)
    ds.no_eval=True; ds.set_num_us(STEPS[-1])
    fo=open("/output/evcue/evcue%s_%s.jsonl"%(TAG,seq),"w"); n=0; t_start=time.time()
    pairs=sorted(recs)
    if MAXP: pairs=pairs[:MAXP]
    for i in pairs:
        R=recs[i]; fr=R.get("frame")
        if fr is None: continue
        anchors={b[0]:b[1:] for b in fr["mb@"+THR]}
        gt0={g[0]:g for g in fr["gt"]}
        if not anchors: continue
        data=ds[i]; t0=int(data.t0); t1=t0+STEPS[-1]
        x=data.pos[:,0].numpy().astype(np.float32); y=data.pos[:,1].numpy().astype(np.float32)
        ta=t1-(1_000_000-data.t.numpy().astype(np.int64))       # absolute event time (us)
        if NOISE_HZ>0:   # v2e-style shot noise: Poisson, uniform over the native 640x480 array, 4 native px per eval px
            nn=RNG.poisson(NOISE_HZ*640*480*(t1-t0)*1e-6*(ds.width*ds.height)/(320*240))
            x=np.concatenate([x,RNG.uniform(0,ds.width,nn).astype(np.float32)]); y=np.concatenate([y,RNG.uniform(0,ds.height,nn).astype(np.float32)])
            ta=np.concatenate([ta,RNG.integers(t0,t1,nn)])
        o=np.argsort(ta,kind="stable"); x,y,ta=x[o],y[o],ta[o]
        # ---- activity indicator, scored on every step (class-agnostic, whole frame halves)
        for d in STEPS:
            q0=time.perf_counter(); a,b=np.searchsorted(ta,[t0+d-WIN,t0+d]); xx=x[a:b]
            nl=int((xx<160).sum()); nr=int((xx>=160).sum()); tact.append((time.perf_counter()-q0)*1e3)
            g=R.get(d,{}).get("gt",[])
            peds=[gg for gg in g if gg[0] in gt0 and (gg[4]-gg[2])>0]
            if peds:
                near=max(peds,key=lambda gg:gg[4]-gg[2]); side_gt=(near[1]+near[3])/2>=160
                act_hits[0]+=int((nr>nl)==side_gt); act_hits[1]+=1
        # ---- event propagation of each anchor box
        state={tid:{"box":list(bx),"c":None} for tid,bx in anchors.items()}
        snap={}
        tt=t0+STEP
        while tt<=t1:
            q0=time.perf_counter(); a,b=np.searchsorted(ta,[tt-WIN,tt]); xx,yy=x[a:b],y[a:b]
            for tid,s in state.items():
                bx=s["box"]; w=bx[2]-bx[0]; h=bx[3]-bx[1]; m=max(3.0,0.25*max(w,h))
                k=inbox(xx,yy,bx,m)
                if k.sum()<MINEV: continue
                cx,cy=float(np.median(xx[k])),float(np.median(yy[k]))
                sx=float(np.median(np.abs(xx[k]-cx)))+1e-3; sy=float(np.median(np.abs(yy[k]-cy)))+1e-3
                if s["c"] is not None:
                    dx,dy=cx-s["c"][0],cy-s["c"][1]
                    s["box"]=[bx[0]+dx,bx[1]+dy,bx[2]+dx,bx[3]+dy]
                    r=np.clip(np.sqrt((sx/s["s"][0])*(sy/s["s"][1])),0.9,1.1)
                    b2=s["box"]; mx,my=(b2[0]+b2[2])/2,(b2[1]+b2[3])/2; hw,hh=(b2[2]-b2[0])/2,(b2[3]-b2[1])/2
                    s["boxsc"]=[mx-hw*r*s.get("acc",1),my-hh*r*s.get("acc",1),mx+hw*r*s.get("acc",1),my+hh*r*s.get("acc",1)]
                    s["acc"]=s.get("acc",1)*r
                s["c"]=(cx,cy); s["s"]=(sx,sy)
            tprop.append((time.perf_counter()-q0)*1e3)
            if (tt-t0) in STEPS: snap[tt-t0]={tid:(list(s["box"]),list(s.get("boxsc",s["box"]))) for tid,s in state.items()}
            tt+=STEP
        xcbox={}
        a0,b0=np.searchsorted(ta,[t0,t0+XWIN]); rx,ry=x[a0:b0],y[a0:b0]
        refs={}
        for tid,bx in anchors.items():
            w=bx[2]-bx[0]; h=bx[3]-bx[1]; m=SEARCH+2
            X0,Y0=int(np.floor(bx[0]))-m,int(np.floor(bx[1]))-m; W,H=int(np.ceil(w))+2*m,int(np.ceil(h))+2*m
            refs[tid]=(X0,Y0,W,H,hist(rx,ry,X0,Y0,W,H))
        for d in STEPS:
            q0=time.perf_counter(); a,b=np.searchsorted(ta,[t0+d-XWIN,t0+d]); cx_,cy_=x[a:b],y[a:b]; xcbox[d]={}
            for tid,bx in anchors.items():
                X0,Y0,W,H,ref=refs[tid]
                if ref.sum()<MINEV or d<=XWIN:
                    xcbox[d][tid]=list(bx); continue
                cur=hist(cx_,cy_,X0,Y0,W,H)
                if cur.sum()<MINEV:
                    xcbox[d][tid]=list(bx); continue
                dx,dy,pk,z=xc_shift(ref,cur)
                if pk-z<0.02: dx,dy=0.0,0.0
                f=d/max(1.0,(d-XWIN))
                dx,dy=dx*f,dy*f
                xcbox[d][tid]=[bx[0]+dx,bx[1]+dy,bx[2]+dx,bx[3]+dy]
            txc.append((time.perf_counter()-q0)*1e3)
        for d in STEPS:
            ev=R.get(d)
            if ev is None: continue
            g={gg[0]:gg for gg in ev["gt"]}; dg={b[0]:b[1:] for b in ev["mb@"+THR]}
            for tid,bx in anchors.items():
                if tid not in g or tid not in gt0: continue
                gb=g[tid][1:]; cls=None
                rec={"seq":seq,"pair":i,"d_ms":d//1000,"tid":tid,"cls":cls_of.get(tid,"?"),
                     "moved":cerr(gt0[tid][1:],gb)*2,
                     "held":[cerr(bx,gb)*2,iou(bx,gb)],
                     "prop":[cerr(snap[d][tid][0],gb)*2,iou(snap[d][tid][0],gb)],
                     "xc":[cerr(xcbox[d][tid],gb)*2,iou(xcbox[d][tid],gb)],
                     "dagr":[cerr(dg[tid],gb)*2,iou(dg[tid],gb)] if tid in dg else None,
                     "h_px":(gb[3]-gb[1])*2}
                fo.write(json.dumps(rec)+"\n"); n+=1
    fo.close(); print("[evcue] %s pairs %d instants %d %.0fs"%(seq,len(pairs),n,time.time()-t_start),flush=True)
print("[evcue] prop per update (all boxes) ms: median %.3f p90 %.3f p99 %.3f n %d"%(np.median(tprop),np.percentile(tprop,90),np.percentile(tprop,99),len(tprop)))
print("[evcue] xc per update (all boxes) ms: median %.3f p90 %.3f p99 %.3f"%(np.median(txc),np.percentile(txc,90),np.percentile(txc,99)))
print("[evcue] activity per update ms: median %.3f p99 %.3f"%(np.median(tact),np.percentile(tact,99)))
print("[evcue] activity side (more events on the side of the nearest labelled pedestrian): %d/%d = %.2f"%(act_hits[0],act_hits[1],act_hits[0]/max(1,act_hits[1])))
