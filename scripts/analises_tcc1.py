"""Análises complementares do TCC 1 (matriz de confusão, validação cruzada e origem do sinal, regra de
sem_followup, recorrência). Saída: results/analises_tcc1.json.

Uso:  python scripts/analises_tcc1.py
"""
import json, sys, warnings, collections
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')
from tcc.ingest import connect, load_conversations
from tcc.dataset import load_task_frame
from tcc.labels import get_labels
from tcc.metrics import bootstrap_ci, macro_f1_fn
from tcc.rules import rule_gargalos
from tcc.segment import conversation_text
from tcc.experiments.tfidf import _make, C_GRID
out={}
conn=connect('data/processed/tcc.db')
df=load_task_frame(conn,'qualificacao','v1'); gold=get_labels(conn,'adjudicado')
convs={c['conversation_id']:c for c in load_conversations(conn)}
L=['qualificado','nao_qualificado']; mf=macro_f1_fn(L)
te=df[df.part=='test'].reset_index(drop=True)
p=pd.read_csv('results/preds_tfidf_qualificacao.csv').set_index('conversation_id').loc[te.conversation_id].pred.tolist()
cm=confusion_matrix(te.y,p,labels=L).tolist()
P,R,F,S=precision_recall_fscore_support(te.y,p,labels=L,zero_division=0)
out['q_test']={'cm[qual,nao]':cm,'acc':accuracy_score(te.y,p),'macroP':P.mean(),'macroR':R.mean(),'macroF':F.mean(),'ci':bootstrap_ci(mf,list(te.y),p,list(te.lead_id),seed=0),'per':{l:[P[i],R[i],F[i],int(S[i])] for i,l in enumerate(L)}}
maj=collections.Counter(df[df.part=='train'].y).most_common(1)[0][0]; pm=[maj]*len(te)
Pm,Rm,Fm,_=precision_recall_fscore_support(te.y,pm,labels=L,zero_division=0)
out['q_majority']={'classe':maj,'acc':accuracy_score(te.y,pm),'macroP':Pm.mean(),'macroR':Rm.mean(),'macroF':Fm.mean()}
def txt(cid,role):
    ms=convs[cid]['messages']
    return conversation_text(ms if role=='all' else [m for m in ms if m['sender_role']==role])
variants={'completo':'all','so_lead':'lead','so_atendente':'atendente'}
def fitbest(X,y,g,seed):
    best=None;bs=-1
    inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    for C in C_GRID:
        s=[]
        for a,b in inner.split(X,y,g):
            m=_make(C,False,seed).fit(X[a],y[a]); s.append(f1_score(y[b],m.predict(X[b]),average='macro',zero_division=0))
        if np.mean(s)>bs: bs=np.mean(s);best=C
    return _make(best,False,seed).fit(X,y),best
# teste fixo por variante (C pela validacao)
out['variantes_teste']={};out['variantes_cv']={}
for nome,role in variants.items():
    d=df.copy(); d['t']=[txt(c,role) for c in d.conversation_id]
    tr,va,ts=(d[d.part==x] for x in('train','val','test'))
    bestC=max(C_GRID,key=lambda C:f1_score(va.y,_make(C,False,0).fit(tr.t,list(tr.y)).predict(va.t),average='macro',zero_division=0))
    pr=list(_make(bestC,False,0).fit(tr.t,list(tr.y)).predict(ts.t))
    out['variantes_teste'][nome]={'C':bestC,'acc':accuracy_score(ts.y,pr),**{k:v for k,v in bootstrap_ci(mf,list(ts.y),pr,list(ts.lead_id),seed=0).items() if k!='n_boot'}}
    X=d.t.values;y=d.y.values;g=d.lead_id.values
    oof=np.empty(len(d),dtype=object);folds=[]
    for a,b in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,g):
        m,_=fitbest(X[a],y[a],g[a],42); oof[b]=m.predict(X[b]); folds.append(f1_score(y[b],oof[b].tolist(),average='macro',zero_division=0))
    out['variantes_cv'][nome]={'folds':folds,'media':float(np.mean(folds)),'dp':float(np.std(folds,ddof=1)),'acc_oof':accuracy_score(y,oof.tolist()),**{k:v for k,v in bootstrap_ci(mf,list(y),oof.tolist(),list(g),seed=0).items() if k!='n_boot'}}
    print(nome,'ok',flush=True)
# objecao agregada
do=load_task_frame(conn,'objecao','v1'); to=do[do.part=='test'].reset_index(drop=True)
po=pd.read_csv('results/preds_tfidf_objecao.csv',keep_default_na=False).set_index('conversation_id').loc[to.conversation_id].pred.tolist()
yb=['com' if len(s)>0 else 'sem' for s in to.y]; pb=['sem' if x in('','nenhuma') else 'com' for x in po]
Lb=['com','sem'];P,R,F,S=precision_recall_fscore_support(yb,pb,labels=Lb,zero_division=0)
out['obj_agregada']={'cm[com,sem]':confusion_matrix(yb,pb,labels=Lb).tolist(),'acc':accuracy_score(yb,pb),'macroP':P.mean(),'macroR':R.mean(),'per':{l:[P[i],R[i],F[i],int(S[i])] for i,l in enumerate(Lb)},'ci':bootstrap_ci(macro_f1_fn(Lb),yb,pb,list(to.lead_id),seed=0)}
majb=collections.Counter('com' if len(s)>0 else 'sem' for s in do[do.part=='train'].y).most_common(1)[0][0]
out['obj_majority']={'classe':majb,'macroF':f1_score(yb,[majb]*len(yb),average='macro',labels=Lb,zero_division=0)}
# regra sem_followup
ids=list(df.conversation_id); G=np.array(['sem_followup' in gold[i]['gargalo'] for i in ids]); Rr=np.array(['sem_followup' in rule_gargalos(convs[i]) for i in ids])
fun=np.array([gold[i]['funil'] for i in ids]); Q=np.isin(fun,['intencao'])
def prf(y,h):
    tp=int((y&h).sum());fp=int((~y&h).sum());fn=int((y&~h).sum());pp=tp/(tp+fp) if tp+fp else 0;rr=tp/(tp+fn) if tp+fn else 0
    return {'tp':tp,'fp':fp,'fn':fn,'p':pp,'r':rr,'f1':2*pp*rr/(pp+rr) if pp+rr else 0}
Gguia=G&Q
out['regra']={'n_gold':int(G.sum()),'gold_intencao':int(Gguia.sum()),'regra_vs_gold':prf(G,Rr),'regra+funil_vs_gold':prf(G,Rr&Q),'regra+funil_vs_gold_guia':prf(Gguia,Rr&Q),'regra_vs_gold_guia':prf(Gguia,Rr),'n_intencao':int(Q.sum()),'dispara':int(Rr.sum()),'dispara_em_intencao':int((Rr&Q).sum())}
# recorrencia
cnt=collections.Counter(c['lead_id'] for c in convs.values()); dist=collections.Counter(cnt.values())
rot=[i for i in ids]; 
out['recorrencia']={'leads':len(cnt),'dist_conversas_por_lead':dict(dist),'leads_recorrentes':sum(v>1 for v in cnt.values()),'conv_retorno_total':sum(1 for c in convs.values() if c['seq_in_lead']>1),'rotuladas_retorno':sum(1 for i in ids if convs[i]['seq_in_lead']>1),'rotuladas_retorno_qualif':sum(1 for i in ids if convs[i]['seq_in_lead']>1 and gold[i]['qualificacao']=='qualificado')}
json.dump(out,open(sys.argv[1],'w',encoding='utf-8'),indent=1,ensure_ascii=False,default=float)
print(json.dumps(out,indent=1,ensure_ascii=False,default=float))
