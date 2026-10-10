"""Eight-subject, three-model development study. No original test EEG loaded."""
import hashlib
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from p300_development import feature, ALPHABET, TRAIN_TEXT
from p300_timing import certify, independent_margin
from openbmi_schedule_audit import fetch, schedule_membership, SCHEDULE_URL, SOURCE_REVISION, BASE

OUT=Path('/kaggle/working/out')
MODELS=['nominal_LDA','shift_augmented_LDA','latency_max_LDA']


def scorer(signal,estimator,latencies,times):
    xs=[feature(signal,int(t)+lag) for t in times for lag in latencies]
    if any(x is None for x in xs):raise ValueError('scorer context outside recorded data')
    return estimator.decision_function(np.stack(xs)).reshape(len(times),len(latencies)).max(1)


def evaluate(signal,estimator,latencies,blocks,truth,subject,model):
    rows=[]
    for char in range(22,33):
        ts,weights=blocks[char]
        symbols=scorer(signal,estimator,latencies,ts)@weights
        candidate=int(np.argmax(symbols));nominal_margin=float(np.sort(symbols)[-1]-np.sort(symbols)[-2])
        prev=blocks[char-1][0][-1];following=blocks[char+1][0][0] if char<32 else len(signal)
        for radius in [2,5,10,15]:
            loc=[];times=[]
            for event,t in enumerate(ts):
                for offset in range(-radius,radius+1):
                    if not prev<t+offset<following:continue
                    if all(feature(signal,t+offset+l) is not None for l in latencies):loc.append((event,offset));times.append(t+offset)
            tables=[{} for _ in ts]
            for (event,offset),score in zip(loc,scorer(signal,estimator,latencies,times)):
                tables[event][offset]=tuple(float(v) for v in score*weights[event])
            nominal_from_cache=np.sum([t[0] for t in tables],axis=0)
            if not np.allclose(nominal_from_cache,symbols,atol=1e-6,rtol=1e-6):raise RuntimeError('nominal cache mismatch')
            for budget in [0,1,3,6]:
                loose=[independent_margin(ts.tolist(),tables,budget,candidate,r) for r in range(36) if r!=candidate]
                coupled=not all(r.exact for r in loose)
                accepted,margins=certify(ts.tolist(),tables,budget,candidate,ordered=coupled)
                witness=None
                if not accepted:
                    for rival,result in margins.items():
                        if result.margin>1e-10:continue
                        shifted=ts+np.asarray(result.offsets)
                        if np.any(np.diff(shifted)<=0) or not prev<shifted[0] or not shifted[-1]<following:raise RuntimeError('infeasible witness')
                        if sum(d!=0 for d in result.offsets)>budget or max(abs(d) for d in result.offsets)>radius:raise RuntimeError('budget violation')
                        cached=np.sum([tables[e][d] for e,d in enumerate(result.offsets)],axis=0)
                        direct=scorer(signal,estimator,latencies,shifted)@weights
                        if not np.allclose(cached,direct,atol=1e-6,rtol=1e-6):raise RuntimeError('raw-window replay mismatch')
                        if int(np.argmax(direct))!=candidate:
                            witness=dict(rival=rival,offsets=list(result.offsets),new_prediction=int(np.argmax(direct)),raw_window_replay_verified=True);break
                rows.append(dict(model=model,subject=subject,character_index=char,truth=int(truth[char]),nominal_prediction=candidate,
                                 nominal_correct=bool(candidate==truth[char]),nominal_margin=nominal_margin,radius_ms=radius*10,budget=budget,
                                 certified=accepted,minimum_pairwise_margin=min(r.margin for r in margins.values()),order_coupling_present=coupled,
                                 independent_certified=all(r.margin>1e-10 for r in loose),exact_witness_decision_change=witness is not None,witness=witness))
            print('CHARACTER_COMPLETE',subject,model,char,radius*10,flush=True)
    return rows


def summarize(rows):
    result=[]
    for model in MODELS:
        for subject in range(1,9):
            for radius in [20,50,100,150]:
                for budget in [0,1,3,6]:
                    subset=[r for r in rows if (r['model'],r['subject'],r['radius_ms'],r['budget'])==(model,subject,radius,budget)]
                    if not subset:continue
                    accepted=[r for r in subset if r['certified']]
                    result.append(dict(model=model,subject=subject,radius_ms=radius,budget=budget,n=len(subset),
                                       clean_accuracy=float(np.mean([r['nominal_correct'] for r in subset])),coverage=len(accepted)/len(subset),
                                       accepted_error=float(np.mean([not r['nominal_correct'] for r in accepted])) if accepted else None,
                                       wrong_accepted_fraction=sum(not r['nominal_correct'] for r in accepted)/len(subset),
                                       verified_decision_change_fraction=float(np.mean([r['exact_witness_decision_change'] for r in subset])),
                                       ordered_extra_acceptances=sum(r['certified'] and not r['independent_certified'] for r in subset)))
    return result


def main():
    OUT.mkdir(exist_ok=True)
    config=dict(protocol='p300-development-comparison-v1',seed=20261010,subjects=list(range(1,9)),session=1,models=MODELS,
                development_only=True,no_original_test_recording_loaded=True,fit_characters=list(range(22)),development_characters=list(range(22,33)),
                shift_ms=[20,50,100,150],budgets=[0,1,3,6],latency_max_samples=[-5,0,5],training_augmentation_samples=[-5,0,5],
                estimator='lsqr shrinkage auto LDA',preprocessing='continuous .5-20Hz zero-phase filter, resample 100Hz, first32 channels, baseline -200:0ms, 8x100ms bins 0:800ms',
                threat='received-centered offline sparse marker shift, strict order, waveform and identities fixed, neighboring characters fixed',
                published_CBLE_reproduced=False,source_revision=SOURCE_REVISION,python=platform.python_version(),
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (OUT/'config.json').write_text(json.dumps(config,indent=2));rows=[];metadata=[]
    with tempfile.TemporaryDirectory(prefix='p300_comparison_') as temporary:
        temp=Path(temporary);schedule=temp/'schedule.mat';fetch(SCHEDULE_URL,schedule)
        membership=schedule_membership(loadmat(schedule)['rc_order']);truth=np.asarray([ALPHABET.index(c) for c in TRAIN_TEXT])
        expected=np.concatenate([np.where(membership[:,c],1,2) for c in truth])
        for subject in range(1,9):
            path=temp/f'sess01_subj{subject:02d}_EEG_ERP.mat';sha=fetch(BASE+f'session1/s{subject}/'+path.name,path)
            d=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
            times=np.asarray(d['t']).reshape(-1);labels=np.asarray(d['y_dec']).reshape(-1);fs=float(np.asarray(d['fs']).reshape(-1)[0])
            if fs!=1000 or len(times)!=1980 or not np.all(times==np.round(times)) or not np.array_equal(labels,expected):raise ValueError('release/schedule/phrase mismatch')
            raw=np.asarray(d['x'])
            signal=resample_poly(sosfiltfilt(butter(4,[.5,20],btype='bandpass',fs=fs,output='sos'),raw[:,:32].astype(np.float64),axis=0),1,10,axis=0).astype(np.float32)
            onsets=np.rint((times-1)/10).astype(int);blocks=[]
            for char in range(33):
                idx=np.arange(char*60,(char+1)*60);idx=idx[np.argsort(onsets[idx],kind='stable')]
                if np.any(np.diff(onsets[idx])<=0):raise ValueError('duplicate resampled markers')
                blocks.append((onsets[idx],membership[idx%60]))
            if any(a[0][-1]>=b[0][0] for a,b in zip(blocks,blocks[1:])):raise ValueError('interleaved characters')
            fit_idx=np.arange(22*60)
            X=[feature(signal,onsets[i]) for i in fit_idx]
            if any(x is None for x in X):raise ValueError('training epoch outside recording')
            nominal=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.stack(X),labels[fit_idx]==1)
            aug_x=[];aug_y=[]
            for i in fit_idx:
                for offset in [-5,0,5]:
                    x=feature(signal,onsets[i]+offset)
                    if x is None:raise ValueError('training augmentation outside recording')
                    aug_x.append(x);aug_y.append(labels[i]==1)
            augmented=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.stack(aug_x),aug_y)
            for model,estimator,latencies in [('nominal_LDA',nominal,[0]),('shift_augmented_LDA',augmented,[0]),('latency_max_LDA',nominal,[-5,0,5])]:
                rows.extend(evaluate(signal,estimator,latencies,blocks,truth,subject,model))
                (OUT/'per_character.json').write_text(json.dumps(rows,indent=2));(OUT/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
            metadata.append(dict(subject=subject,file_sha256=sha,phrase_schedule_verified=True,minimum_gap_ms=int(min(np.diff(b[0]).min() for b in blocks)*10)))
            (OUT/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2));del d,raw,signal;path.unlink()
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True,publication_ready=False,scope='eight-subject development comparison; no test EEG accessed')))
    print('COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':main()
