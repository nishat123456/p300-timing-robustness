"""OpenBMI marker robustness development pilot; original train recordings only.

Own shrinkage-LDA decoder, not the original toolbox reproduction. Test recordings
are not requested from loadmat. Raw EEG stays in temporary cloud storage.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from p300_timing import certify, independent_margin
from openbmi_schedule_audit import fetch, schedule_membership, SCHEDULE_URL, SOURCE_REVISION, BASE

OUT=Path('/kaggle/working/out')
SEED=20261010
ALPHABET='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'
TRAIN_TEXT='NEURAL_NETWORKS_AND_DEEP_LEARNING'


def feature(signal,onset):
    # All bounds in resampled 100 Hz coordinates; no padding or wrapping.
    if onset-20<0 or onset+80>len(signal):return None
    baseline=signal[onset-20:onset].mean(0)
    epoch=signal[onset:onset+80]-baseline
    return epoch.reshape(8,10,32).mean(1).reshape(-1)


def main():
    OUT.mkdir(exist_ok=True)
    config=dict(seed=SEED,source_revision=SOURCE_REVISION,subjects=[1,2],session=1,
                data_boundary='EEG_ERP_train only; first 22 characters fit, last 11 development',
                sampling_hz=100,channels=list(range(1,33)),band_hz=[.5,20],window_ms=[0,800],baseline_ms=[-200,0],
                estimator='LDA lsqr auto shrinkage; eight 100ms mean bins/channel',
                shift_ms=[20,50,100,150],change_budgets=[0,1,3,6],threat='offline marker-only integer shifts; waveform and flash membership fixed; strict chronological order; other characters fixed',
                no_test_recording_loaded=True,received_centered_invariance_not_correctness=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                timing_sha256=hashlib.sha256((Path(__file__).parent/'p300_timing.py').read_bytes()).hexdigest())
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    records=[];metadata=[]
    with tempfile.TemporaryDirectory(prefix='p300_development_') as folder:
        folder=Path(folder);schedule_path=folder/'schedule.mat';fetch(SCHEDULE_URL,schedule_path)
        membership=schedule_membership(loadmat(schedule_path)['rc_order'])
        for subject in [1,2]:
            path=folder/f'sess01_subj{subject:02d}_EEG_ERP.mat'
            digest=fetch(BASE+f'session1/s{subject}/'+path.name,path)
            data=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
            times=np.asarray(data['t']).reshape(-1);labels=np.asarray(data['y_dec']).reshape(-1)
            fs=float(np.asarray(data['fs']).reshape(-1)[0])
            if fs!=1000 or len(times)!=33*60 or not np.all(times==np.round(times)):
                raise ValueError('unexpected development release structure')
            # Independent known phrase check, not merely deriving character IDs from labels.
            target=np.array([ALPHABET.index(c) for c in TRAIN_TEXT])
            expected=np.concatenate([np.where(membership[:,c],1,2) for c in target])
            if not np.array_equal(labels,expected):raise ValueError('published phrase/schedule mismatch')
            raw=np.asarray(data['x'])
            if raw.ndim!=2 or raw.shape[1]<32:raise ValueError('unexpected time-major EEG array')
            signal=resample_poly(sosfiltfilt(butter(4,[.5,20],btype='bandpass',fs=fs,output='sos'),raw[:,:32].astype(np.float64),axis=0),1,10,axis=0).astype(np.float32)
            # Source segmentation uses one-based MATLAB t, converted before resampling.
            onsets=np.rint((times-1)/10).astype(int)
            blocks=[]
            for char in range(33):
                idx=np.arange(char*60,(char+1)*60);idx=idx[np.argsort(onsets[idx],kind='stable')]
                ts=onsets[idx]
                if np.any(np.diff(ts)<=0):raise ValueError('duplicate/reversed downsampled onsets')
                blocks.append((idx,ts,membership[idx%60]))
            for left,right in zip(blocks,blocks[1:]):
                if left[1][-1]>=right[1][0]:raise ValueError('interleaved character intervals')
            fit_idx=np.concatenate([b[0] for b in blocks[:22]])
            X=[feature(signal,onsets[i]) for i in fit_idx]
            if any(x is None for x in X):raise ValueError('nominal training epoch out of bounds')
            lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.stack(X),labels[fit_idx]==1)
            metadata.append(dict(subject=subject,file_sha256=digest,character_schedule_verified=True,fit_characters=22,development_characters=11,
                                 minimum_flash_gap_ms=int(min(np.diff(b[1]).min() for b in blocks)*10),index_origin='public MATLAB t convention, one-based converted'))
            for char in range(22,33):
                idx,ts,weights=blocks[char]
                nominal=np.stack([feature(signal,t) for t in ts]);scores=lda.decision_function(nominal)
                symbol=scores@weights;candidate=int(np.argmax(symbol));truth=int(target[char])
                for radius in [2,5,10,15]:
                    features=[];locations=[]
                    prev=blocks[char-1][1][-1] if char else -1
                    following=blocks[char+1][1][0] if char<32 else len(signal)
                    for event,t in enumerate(ts):
                        for offset in range(-radius,radius+1):
                            if not prev<t+offset<following:continue
                            x=feature(signal,t+offset)
                            if x is not None:features.append(x);locations.append((event,offset))
                    contribution=[{} for _ in ts]
                    for (event,offset),score in zip(locations,lda.decision_function(np.stack(features))):
                        contribution[event][offset]=tuple(float(v) for v in score*weights[event])
                    for budget in [0,1,3,6]:
                        loose=[independent_margin(ts.tolist(),contribution,budget,candidate,r) for r in range(36) if r!=candidate]
                        need_order=not all(v.exact for v in loose)
                        certified,margins=certify(ts.tolist(),contribution,budget,candidate,ordered=need_order)
                        attacked=False;witness=None
                        if not certified:
                            # Verify a full multiclass decision, not just a negative rival margin.
                            for rival,result in margins.items():
                                if result.margin>1e-10:continue
                                changed=np.sum([contribution[e][d] for e,d in enumerate(result.offsets)],axis=0)
                                if int(np.argmax(changed))!=candidate:
                                    attacked=True;witness=dict(rival=rival,offsets=list(result.offsets),new_prediction=int(np.argmax(changed)));break
                        records.append(dict(subject=subject,character_index=char,truth=truth,nominal_prediction=candidate,nominal_correct=candidate==truth,
                                            radius_ms=radius*10,budget=budget,certified=certified,minimum_pairwise_margin=min(r.margin for r in margins.values()),
                                            order_coupling_present=need_order,independent_certified=all(v.margin>1e-10 for v in loose),
                                            exact_witness_decision_change=attacked,witness=witness))
                    print('CHARACTER_COMPLETE',subject,char,radius*10,flush=True)
                    (OUT/'per_character.json').write_text(json.dumps(records,indent=2))
            del data,raw,signal;path.unlink()
            (OUT/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2))
    summaries=[]
    for subject in [1,2]:
        for radius in [20,50,100,150]:
            for budget in [0,1,3,6]:
                subset=[r for r in records if r['subject']==subject and r['radius_ms']==radius and r['budget']==budget]
                accepted=[r for r in subset if r['certified']]
                summaries.append(dict(subject=subject,radius_ms=radius,budget=budget,n=len(subset),clean_accuracy=float(np.mean([r['nominal_correct'] for r in subset])),
                                      coverage=len(accepted)/len(subset),accepted_error=float(np.mean([not r['nominal_correct'] for r in accepted])) if accepted else None,
                                      wrong_accepted_fraction=sum(not r['nominal_correct'] for r in accepted)/len(subset),
                                      verified_decision_change_fraction=float(np.mean([r['exact_witness_decision_change'] for r in subset])),
                                      ordered_extra_acceptances=sum(r['certified'] and not r['independent_certified'] for r in subset)))
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True,scope='two-subject development only; not final test, generalization or publication readiness')))
    print('P300_DEVELOPMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
