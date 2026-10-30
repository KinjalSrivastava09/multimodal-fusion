#!/usr/bin/env python3
"""Self-contained reproducible synthetic image--text fusion experiment.
Usage: python project.py   |   python project.py --check
No external datasets, pretrained checkpoints or network access.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parent
SEED=20261030
VOCAB=['horizontal','vertical','possible','present','suggests','no','definite','pattern','opacity','involvement']

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(2)

def make_data(n=1800):
    """Two independent latent bits define XOR target; weak per-modality label hints.
    Image: band indicates noisy label, square location indicates bit a.
    Text: orientation indicates noisy label, opacity/involvement indicates bit b.
    Thus joint model can disambiguate independent unimodal uncertainty.
    """
    rng=np.random.default_rng(SEED)
    a=rng.integers(0,2,n); b=rng.integers(0,2,n); y=(a^b).astype(np.int64)
    image_hint=np.where(rng.random(n)<.83,y,1-y)
    text_hint=np.where(rng.random(n)<.76,y,1-y)
    X=np.empty((n,1,24,24),dtype='float32'); T=np.zeros((n,10),dtype='float32'); sentences=[]
    for i in range(n):
        im=rng.normal(0,.17,(24,24)).astype('float32')
        if image_hint[i]==0: im[10:13,2:22]+=.72
        else: im[2:22,10:13]+=.72
        if a[i]==0: im[17:22,2:7]+=.94
        else: im[17:22,17:22]+=.94
        X[i,0]=np.clip(im,0,1)
        orient='horizontal' if text_hint[i]==0 else 'vertical'
        word='opacity' if b[i]==0 else 'involvement'
        mod=['present','possible','definite','suggests'][int(rng.integers(4))]
        # Keep a 10-feature bag-of-words as in source proposal.
        words=[orient,'pattern',mod,word]
        if mod=='possible' and rng.random()<.5: words.append('no')
        sentences.append(' '.join(words))
        for token in words: T[i,VOCAB.index(token)]=1
    return X,T,y,a,b,image_hint,text_hint,sentences

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(16*6*6,16),nn.ReLU())
    def forward(self,x): return self.net(x)
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(10,16),nn.ReLU())
    def forward(self,t): return self.net(t)
class Net(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        self.image=ImageEncoder() if mode!='text' else None
        self.text=TextEncoder() if mode!='image' else None
        self.head=nn.Sequential(nn.Linear(32 if mode=='fusion' else 16,24),nn.ReLU(),nn.Linear(24,2))
    def embed(self,x,t,ablate=False):
        if self.mode=='image': return self.image(x)
        if self.mode=='text': return self.text(t)
        it=self.image(x); tt=self.text(t)
        if ablate: tt=torch.zeros_like(tt)
        return torch.cat([it,tt],dim=1)
    def forward(self,x,t,ablate=False): return self.head(self.embed(x,t,ablate))

def evaluate(model,X,T,y,ablate=False):
    model.eval()
    with torch.no_grad():
        z=model(X,T,ablate); pred=z.argmax(1).numpy(); loss=nn.functional.cross_entropy(z,torch.from_numpy(y)).item()
    cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'loss':float(loss),'precision_macro':float(p),'recall_macro':float(r),'f1_macro':float(f),'confusion_matrix':cm,'predictions':pred.tolist()}

def fit(mode,tr,va):
    seed_all(); model=Net(mode); optim=torch.optim.Adam(model.parameters(),lr=.003)
    ds=TensorDataset(torch.from_numpy(tr[0]),torch.from_numpy(tr[1]),torch.from_numpy(tr[2]))
    loader=DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    curves={'train_loss':[],'val_loss':[],'val_accuracy':[]}
    for epoch in range(12):
        model.train(); total=0
        for x,t,y in loader:
            optim.zero_grad(); loss=nn.functional.cross_entropy(model(x,t),y); loss.backward();optim.step();total+=loss.item()*len(y)
        ev=evaluate(model,torch.from_numpy(va[0]),torch.from_numpy(va[1]),va[2])
        curves['train_loss'].append(total/len(ds));curves['val_loss'].append(ev['loss']);curves['val_accuracy'].append(ev['accuracy'])
    return model,curves

def savefig(name):
    plt.savefig(ROOT/'figures'/name,bbox_inches='tight',dpi=180);plt.close()

def plot_samples(X,y,sentences):
    fig,ax=plt.subplots(2,4,figsize=(10,5.5));
    for j,axis in enumerate(ax.flat):
        axis.imshow(X[j,0],cmap='gray',vmin=0,vmax=1);axis.set_title(f'Class {y[j]}',fontsize=10)
        axis.set_xlabel(sentences[j].replace(' pattern',''),fontsize=7);axis.set_xticks([]);axis.set_yticks([])
    fig.suptitle('Generated paired examples (each caption is its paired text)',fontsize=12);fig.tight_layout();savefig('paired_examples.png')

def plot_arch():
    fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,4)
    boxes=[(.4,2.7,'24×24 image'),(2.5,2.7,'CNN → 16'),(.4,.7,'10-word vector'),(2.5,.7,'Linear → 16'),(5,1.7,'Concat → 32'),(7.2,1.7,'MLP → 24'),(9,1.7,'2 logits')]
    for x,y,s in boxes: ax.text(x,y,s,ha='center',va='center',bbox={'boxstyle':'round,pad=.45','fc':'#eaf1f8','ec':'#4f6680'},fontsize=9)
    for start,end in [((1,2.7),(1.8,2.7)),((1,.7),(1.8,.7)),((3.25,2.7),(4.35,1.85)),((3.25,.7),(4.35,1.55)),((5.8,1.7),(6.5,1.7)),((8.1,1.7),(8.4,1.7))]: ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.3})
    savefig('architecture.png')

def plot_results(curves,metrics,emb,y):
    fig,ax=plt.subplots(figsize=(7,4));
    for k,v in curves.items(): ax.plot(range(1,13),v['train_loss'],marker='.',label=k)
    ax.set(xlabel='Epoch',ylabel='Observed training cross-entropy',title='Training loss (measured)');ax.legend();ax.grid(alpha=.2);savefig('training_loss.png')
    fig,ax=plt.subplots(figsize=(7,4)); keys=['image','text','fusion','fusion_no_text']; labels=['Image only','Text only','Fusion','Fusion / text zeroed']; vals=[metrics[k]['accuracy']*100 for k in keys]
    bars=ax.bar(labels,vals);ax.set_ylim(0,105);ax.set_ylabel('Test accuracy (%)');ax.set_title('Held-out test performance');ax.tick_params(axis='x',labelrotation=15)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}%',ha='center',fontsize=9)
    savefig('accuracy.png')
    cm=np.array(metrics['fusion']['confusion_matrix']);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=14)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Class 0','Class 1'],yticklabels=['Class 0','Class 1'],xlabel='Predicted',ylabel='True',title='Fusion test confusion matrix');savefig('confusion_matrix.png')
    xy=PCA(n_components=2,random_state=SEED).fit_transform(emb);fig,ax=plt.subplots(figsize=(6.5,4.5));
    for k in [0,1]: ax.scatter(xy[y==k,0],xy[y==k,1],s=16,alpha=.55,label=f'Class {k}')
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title='PCA of actual test fusion embeddings');ax.legend();savefig('latent_pca.png')

def run():
    seed_all();X,T,y,a,b,ih,th,sentences=make_data()
    rng=np.random.default_rng(SEED+1); perm=rng.permutation(len(y)); train,valid,test=perm[:1260],perm[1260:1350],perm[1350:]
    subset=lambda ix:(X[ix],T[ix],y[ix])
    tr,va,te=subset(train),subset(valid),subset(test)
    assert len(set(train)&set(test))==0 and len(set(valid)&set(test))==0
    assert np.array_equal(y,a^b) and len(VOCAB)==10
    metrics={};curves={};trained={}
    for mode in ['image','text','fusion']:
        model,cur=fit(mode,tr,va); trained[mode]=model;curves[mode]=cur
        metrics[mode]=evaluate(model,torch.from_numpy(te[0]),torch.from_numpy(te[1]),te[2]);print(mode,metrics[mode]['accuracy'],flush=True)
    metrics['fusion_no_text']=evaluate(trained['fusion'],torch.from_numpy(te[0]),torch.from_numpy(te[1]),te[2],True)
    plot_samples(X,y,sentences);plot_arch()
    with torch.no_grad(): embeds=trained['fusion'].embed(torch.from_numpy(te[0]),torch.from_numpy(te[1])).numpy()
    plot_results(curves,metrics,embeds,te[2])
    payload={'seed':SEED,'sample_count':1800,'split_sizes':{'train':1260,'validation':90,'test':450},'test_indices':test.tolist(),'test_labels':te[2].tolist(),'vocabulary':VOCAB,'generation':{'image_hint_correct_probability':.83,'text_hint_correct_probability':.76,'target':'a XOR b'},'epochs':12,'learning_rate':.003,'batch_size':64,'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__},'models':metrics,'curves':curves}
    (ROOT/'results'/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
    for name,model in trained.items():torch.save(model.state_dict(),ROOT/'results'/f'{name}_weights.pt')
    check()
    print('\nHELD-OUT TEST RESULTS (n=450)')
    for key,label in [('image','Image-only'),('text','Text-only'),('fusion','Multimodal fusion'),('fusion_no_text','Fusion: text embedding zeroed')]:
        result=metrics[key]
        print(f'{label:30} {result["accuracy"]*100:6.2f}%  |  F1 (macro): {result["f1_macro"]:.3f}')
    print(f'\nSaved reproducible metrics to {ROOT / "results" / "metrics.json"}')
    print(f'Saved six figures to {ROOT / "figures"}')

def check():
    q=json.loads((ROOT/'results'/'metrics.json').read_text());y=np.array(q['test_labels']);assert len(y)==450
    assert sorted(q['split_sizes'].values())==[90,450,1260]
    for k,v in q['models'].items():
        p=np.array(v['predictions']); assert len(p)==len(y)
        assert abs((p==y).mean()-v['accuracy'])<1e-12
        assert confusion_matrix(y,p,labels=[0,1]).tolist()==v['confusion_matrix']
    assert len(list((ROOT/'figures').glob('*.png')))==6
    assert all(len(c['train_loss'])==12 for c in q['curves'].values())
    print('PASS: saved predictions, accuracies, confusion matrices, splits, figures and epoch logs')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args()
    if args.check:check()
    else:run()
