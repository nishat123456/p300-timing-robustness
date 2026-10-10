"""Cloud-only schedule compatibility audit, fixed subject 1, both sessions.

No decoding, detector fitting, new splits, or BNCI2015-003 access. Candidate
stored/chronological orderings are diagnostics; neither becomes a loader unless
schedule and independent character identity are validated. No EEG outputs.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import requests
from scipy.io import loadmat

SOURCE_REVISION = 'c0e56825366681ae576801af08eb5fe326a5afd8'
SCHEDULE_URL = f'https://raw.githubusercontent.com/PatternRecognition/OpenBMI/{SOURCE_REVISION}/GigaScience/function_ERP/random_cell_order.mat'
BASE = 'https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/'


def schedule_membership(cells):
    arrays = [np.asarray(cell) for cell in np.asarray(cells,dtype=object).ravel()]
    if len(arrays) != 10 or any(a.shape != (12,6) for a in arrays):
        raise ValueError('expected published ten 12x6 schedule cells; refuse guessed adaptation')
    # Original ERP_performance uses Nsequence=5: first five cells, not all ten.
    schedule = np.concatenate(arrays[:5],axis=0)
    if not np.isfinite(schedule.astype(float)).all() or not np.all(schedule == np.round(schedule)) or schedule.min() != 1 or schedule.max() != 36:
        raise ValueError('schedule IDs must be integers 1..36')
    if any(len(set(row)) != 6 for row in schedule):
        raise ValueError('each flash must contain six distinct symbols')
    membership = np.zeros((60,36),dtype=bool)
    for i,row in enumerate(schedule): membership[i,np.asarray(row,dtype=int)-1] = True
    if not np.all(membership.sum(axis=0) == 10):
        raise ValueError('each candidate must occur ten times in five sequences')
    return membership


def alignment(labels, membership):
    labels = np.asarray(labels).ravel()
    if not np.isin(labels,[1,2]).all() or len(labels)%60:
        return dict(status='incompatible_label_vocab_or_count')
    target = labels.reshape(-1,60) == 1
    distance = np.sum(target[:,:,None] != membership[None,:,:],axis=1)
    exact_count = (distance == 0).sum(axis=1)
    return dict(n_blocks=len(target),unique_exact_blocks=int((exact_count==1).sum()),
                no_exact_blocks=int((exact_count==0).sum()),ambiguous_blocks=int((exact_count>1).sum()),
                min_mismatch_per_block=distance.min(axis=1).astype(int).tolist(),
                # Candidate identities are supported only for exact unique matches.
                exact_candidate_ids=[int(np.flatnonzero(row==0)[0]+1) if (row==0).sum()==1 else None for row in distance],
                all_unique_exact=bool(np.all(exact_count==1)))


def fetch(url,path):
    digest=hashlib.sha256()
    with requests.get(url,stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        with path.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                if chunk:stream.write(chunk);digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20261010)
    args=parser.parse_args()
    out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='openbmi_schedule_') as folder:
        temp=Path(folder)
        path=temp/'schedule.mat'
        schedule_sha=fetch(SCHEDULE_URL,path)
        mat=loadmat(path)
        keys=[k for k in mat if not k.startswith('__')]
        if len(keys)!=1: raise ValueError('unexpected public schedule schema')
        print('SCHEDULE_SCHEMA', keys[0], mat[keys[0]].shape, mat[keys[0]].dtype, [getattr(x,'shape',None) for x in mat[keys[0]].ravel()],flush=True)
        membership=schedule_membership(mat[keys[0]])
        # Focused functional checks on synthetic label patterns, entirely cloud.
        valid=np.where(membership[:,0],1,2)
        assert alignment(valid,membership)['all_unique_exact']
        invalid=valid.copy();invalid[0]=3-invalid[0]
        assert not alignment(invalid,membership)['all_unique_exact']
        assert alignment([3]*60,membership)['status']=='incompatible_label_vocab_or_count'
        assert alignment(valid[:59],membership)['status']=='incompatible_label_vocab_or_count'
        assert membership.shape==(60,36)
        (out/'focused_tests.json').write_text(json.dumps(dict(checks_passed=5,scope='schedule/parser synthetic checks; not full repository suite')))
        config=dict(seed=args.seed,source_revision=SOURCE_REVISION,schedule_url=SCHEDULE_URL,schedule_sha256=schedule_sha,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),subject=1,sessions=[1,2],
                    ordering_candidates=['stored','stable_chronological'],performance_evaluated=False,
                    independent_character_groundtruth_verified=False,BNCI2015_003_accessed=False,
                    index_origin_evidence='public prep_segmentation indexes MATLAB x at t, suggesting 1-based convention; release alignment still requires audit')
        (out/'config.json').write_text(json.dumps(config,indent=2))
        for session in (1,2):
            name=f'sess{session:02d}_subj01_EEG_ERP.mat'
            path=temp/name
            url=BASE+f'session{session}/s1/'+name
            print('Cloud metadata audit',session,flush=True)
            sha=fetch(url,path)
            data=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train','EEG_ERP_test'])
            for split in ('train','test'):
                d=data['EEG_ERP_'+split]
                times=np.asarray(d['t']).ravel()
                labels=np.asarray(d['y_dec']).ravel()
                if len(times)!=len(labels): raise ValueError('unaligned event vectors')
                order=np.argsort(times,kind='stable')
                result=dict(session=session,split=split,file_sha256=sha,file_bytes=path.stat().st_size,
                            fields=list(d),n_samples=int(np.asarray(d['x']).shape[0]),n_events=len(times),
                            integer_onsets=bool(np.all(times==np.round(times))),duplicate_onsets=int(len(times)-len(np.unique(times))),
                            stored=alignment(labels,membership),chronological=alignment(labels[order],membership),
                            status='candidate schedule check only; independent symbols/character boundaries still need external validation')
                records.append(result)
            del data
            path.unlink()
            (out/'schedule_alignment.json').write_text(json.dumps(records,indent=2))
    (out/'README.md').write_text('Metadata-only schedule alignment audit. No EEG, scores, predictions, attacks or detector outcomes exported. Candidate matches are necessary but insufficient for release-compatible character identity. No new loader or confirmation performance claim. Frozen splits/gates unchanged.\n')
    print('SCHEDULE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
