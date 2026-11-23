"""Reproduce web figures from the actual trajectory and original checkpoint diagnostics, never fixtures."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True,help='Directory containing the Colab result files')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]; out=root/'assets';out.mkdir(exist_ok=True)
src=args.data
z=np.load(src/'state_snapshots.npz',allow_pickle=False)
def rows(name):
 with (src/name).open() as f: return list(csv.DictReader(f))
m=rows('state_metrics.csv');q=rows('static_quantization.csv')
for row in m+q:
 for key in row:
  if key=='grouping': continue
  row[key]=int(row[key]) if key in ['tokens','layer','head','bits','groups','r90'] else float(row[key])
meta=json.loads((src/'run_metadata.json').read_text())
times=[64,128,192,256]; layers=[0,8,16]; heads=[0,1]
states=np.stack([np.stack([np.stack([z[f't{t:05d}_l{l:02d}_h{h:02d}'] for h in heads]) for l in layers]) for t in times]).astype('<f4')
assert states.shape==(4,3,2,128,128) and np.isfinite(states).all()
trajectory=np.load(src/'recurrent_state_trajectory.npz',allow_pickle=False)
full_states=trajectory['states']
assert full_states.shape==(256,3,2,128,128) and full_states.dtype==np.float32
assert np.isfinite(full_states).all()
state_files={}
for li,l in enumerate(layers):
 for hi,h in enumerate(heads):
  name=f'states-L{l}-H{h}.f32'
  full_states[:,li,hi].astype('<f4').tofile(out/name)
  state_files[f'L{l}-H{h}']=name
for ti,t in enumerate(times):np.testing.assert_array_equal(states[ti],full_states[t-1])
old=out/'states.f32'
if old.exists():old.unlink()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':12,'axes.titleweight':'medium','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#cbd5e1','axes.labelcolor':'#344256','xtick.color':'#526175','ytick.color':'#526175','text.color':'#172b4d','grid.color':'#e5eaf0','svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white','savefig.bbox':'tight','savefig.pad_inches':.15})
colors={0:'#243b63',8:'#3b82c4',16:'#148579'}
keycolor='#2877b8'; valcolor='#d47a36'
artifacts=[]
def save(fig,stem):
 for ext in ['svg','png']:
  fig.savefig(out/f'{stem}.{ext}',dpi=180)
 plt.close(fig);artifacts.append(stem)
def grid(ax): ax.grid(axis='y',alpha=.7);ax.set_axisbelow(True)
def structure_axes(axs,x,l):
 u,s,vh=np.linalg.svd(x.astype(float),full_matrices=False)
 e=np.cumsum(s*s)/(s*s).sum(); rr=np.sqrt(np.mean(x.astype(float)**2,axis=1));cc=np.sqrt(np.mean(x.astype(float)**2,axis=0))
 im=axs[0].imshow(abs(x),origin='lower',aspect='equal',cmap='magma',interpolation='nearest')
 axs[0].set(xlabel='Value channel',ylabel='Key channel',title='State magnitude |S|',xticks=[0,32,64,96,127],yticks=[0,32,64,96,127]);axs[0].figure.colorbar(im,ax=axs[0],shrink=.77,pad=.03)
 axs[1].plot(rr,color=keycolor,lw=1.7,label='Key / row');axs[1].plot(cc,color=valcolor,lw=1.7,label='Value / column');axs[1].set(xlabel='Channel index',ylabel='RMS',title='Channel scale',xlim=(0,127));axs[1].legend(fontsize=9,frameon=False);grid(axs[1])
 axs[2].plot(np.arange(1,129),100*e,color=colors[l],lw=2);axs[2].axhline(90,color='#a8b2c2',lw=1,ls='--');axs[2].set(xlabel='Retained rank r',ylabel='Energy retained (%)',title='Singular spectrum',xlim=(1,32),ylim=(65,101));axs[2].set_xticks([1,8,16,24,32]);grid(axs[2]);axs[2].annotate(f'E₁ = {e[0]*100:.2f}%\nr₉₀ = {int(np.searchsorted(e,.9)+1)}',xy=(.42,.22),xycoords='axes fraction',fontsize=11,color=colors[l])
 return u,s,vh,rr,cc,e
summ=[]
audit=[]
for li,l in enumerate(layers):
 x=states[-1,li,0]
 fig,axs=plt.subplots(1,3,figsize=(13.8,3.65),layout='constrained')
 u,s,vh,rr,cc,e=structure_axes(axs,x,l)
 save(fig,f'structure-l{l}')
 for part in ['magnitude','channels','spectrum']:
  fig,ax=plt.subplots(figsize=(5.3,4.0),layout='constrained')
  if part=='magnitude':
   im=ax.imshow(abs(x),origin='lower',cmap='magma',interpolation='nearest');ax.set(xlabel='Value channel',ylabel='Key channel',xticks=[0,32,64,96,127],yticks=[0,32,64,96,127]);fig.colorbar(im,ax=ax,pad=.03,label='|State entry|')
  elif part=='channels':
   ax.plot(rr,color=keycolor,label='Key / row',lw=1.7);ax.plot(cc,color=valcolor,label='Value / column',lw=1.7);ax.set(xlabel='Channel index',ylabel='RMS',xlim=(0,127));ax.legend(frameon=False);grid(ax)
  else:
   ax.plot(np.arange(1,129),100*e,color=colors[l],lw=2);ax.axhline(90,color='#a8b2c2',ls='--',lw=1);ax.set(xlabel='Retained rank r',ylabel='Energy retained (%)',xlim=(1,32),ylim=(65,101),xticks=[1,8,16,24,32]);grid(ax)
  ax.set_title(f'Layer {l} · Head 0 · t = 256',loc='left',pad=15)
  save(fig,f'layer-{l}-'+{'magnitude':'heatmap','channels':'rms','spectrum':'spectrum'}[part])
 row=next(r for r in m if r['tokens']==256 and r['layer']==l and r['head']==0)
 summ.append(dict(layer=l,head=0,tokens=256,E1=row['E1'],E8=row['E8'],r90=row['r90'],row_peak_channel=int(rr.argmax()),column_peak_channel=int(cc.argmax()),row_peak_rms=float(rr.max()),column_peak_rms=float(cc.max()),max_over_rms=row['max_over_rms'],u1_ipr=row['u1_ipr'],v1_ipr=row['v1_ipr']))
for r in m:
 x=z[f"t{r['tokens']:05d}_l{r['layer']:02d}_h{r['head']:02d}"].astype(float);u,s,vh=np.linalg.svd(x,full_matrices=False);energy=s*s;retained=np.cumsum(energy)/energy.sum()
 checks={'E1':retained[0],'E8':retained[7],'r90':np.searchsorted(retained,.9)+1,'max_over_rms':np.abs(x).max()/np.sqrt(np.mean(x*x)),'u1_ipr':np.sum(u[:,0]**4),'v1_ipr':np.sum(vh[0]**4)}
 for k,v in checks.items():
  if not np.isclose(v,r[k],rtol=1e-8,atol=1e-9): raise AssertionError((r,k,v))
 audit.append({'tokens':r['tokens'],'layer':r['layer'],'head':r['head'],'metrics_recomputed_match':True})
for precision in [4,8]:
 fig,axs=plt.subplots(1,3,figsize=(12.8,4.0),sharey=True,layout='constrained')
 for ax,l in zip(axs,layers):
  vals=[next(r['relative_frobenius_error'] for r in q if r['tokens']==256 and r['layer']==l and r['head']==0 and r['bits']==precision and r['grouping']==g) for g in ['tensor','row','column']]
  bars=ax.bar(['Tensor','Row','Column'],vals,color=['#9aa7b6',keycolor,valcolor],width=.6)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+(.035 if precision==4 else .0015),f'{v:.3f}',ha='center',va='bottom',fontsize=11)
  ax.set(title=f'Layer {l} · Head 0',ylim=(0,2.35 if precision==4 else .14));grid(ax)
 axs[0].set_ylabel('Relative Frobenius error')
 save(fig,f'quantization-{precision}bit')
# Fully specified zero-point diagnostic (separate from the original experiment).
x=z['t00256_l08_h00'].astype(float);scale=float((x.max()-x.min())/15);zp=int(np.clip(np.round(-x.min()/scale),0,15));codes=np.clip(np.round(x/scale)+zp,0,15);reconstructed=(codes-zp)*scale
zero_diag={'scope':'Additional offline diagnostic on the same L8/H0/t256 state; not a new model run','bits':4,'layer':8,'head':0,'tokens':256,'formula':'scale=(max-min)/(2^b-1); zp=clip(round(-min/scale),0,2^b-1); q=clip(round(S/scale)+zp,0,2^b-1); Shat=(q-zp)*scale','rounding':'NumPy round: ties to even','scale':scale,'zero_point':zp,'relative_frobenius_error':float(np.linalg.norm(x-reconstructed)/np.linalg.norm(x)),'original_relative_frobenius_error':next(r['relative_frobenius_error'] for r in q if r['tokens']==256 and r['layer']==8 and r['head']==0 and r['bits']==4 and r['grouping']=='tensor')}
for r in q:
 x=z[f"t{r['tokens']:05d}_l{r['layer']:02d}_h{r['head']:02d}"].astype(float);axis={'tensor':None,'row':1,'column':0}[r['grouping']];lo=x.min(axis=axis,keepdims=True);hi=x.max(axis=axis,keepdims=True);scale=(hi-lo)/(2**r['bits']-1);safe=np.where(scale==0,1,scale);rec=np.clip(np.rint((x-lo)/safe),0,2**r['bits']-1)*safe+lo;err=np.linalg.norm(rec-x)/(np.linalg.norm(x)+1e-12)
 assert np.isclose(err,r['relative_frobenius_error'],rtol=1e-8,atol=1e-9)
# Provenance and data for browser. All axis IDs are explicit.
payload={'schema_version':1,'state_files':state_files,'state_dtype':'float32','state_endianness':'little','state_shape':list(full_states.shape),'state_axes':['time','layer','head','key_channel','value_channel'],'checkpoint_tokens':times,'token_indices':trajectory['token_indices'].tolist(),'token_ids':trajectory['token_ids'].tolist(),'token_texts':trajectory['token_texts'].tolist(),'layer_ids':layers,'head_ids':heads,'snapshots':1536,'analysis_snapshots':24,'capture':'Every token recorded after processing: 256 actual time steps. Temporal SVD diagnostics use every token for head 0; original checkpoint and quantization diagnostics use four checkpoints. No synthetic frames.','temporal_diagnostics_file':'temporal_metrics.json','temporal_analysis_snapshots':768,'temporal_head_ids':[0],'metadata':meta,'summary_head0_t256':summ,'metrics':m,'quantization':q,'zero_point_diagnostic':zero_diag,'pair_max_abs':np.max(np.abs(full_states),axis=(0,3,4)).tolist(),'global_max_abs':float(np.abs(full_states).max())}
(out/'experiment.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
(out/'numeric_audit.json').write_text(json.dumps({'snapshot_count':1536,'analysis_snapshot_count':24,'analysis_checkpoints_equal_trajectory':True,'finite_fp32_shape_validated':True,'all24_metric_rows_match':True,'all144_quantization_rows_match':True,'source_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['recurrent_state_trajectory.npz','state_snapshots.npz','state_metrics.csv','static_quantization.csv','run_metadata.json']},'summary':summ,'zero_point_diagnostic':zero_diag,'checks':audit},indent=2))
for name in ['state_snapshots.npz','state_metrics.csv','static_quantization.csv','run_metadata.json','validation_comparisons.csv','validation_report.json']:
 shutil.copy2(src/name,out/name)
# Recompute temporal curves at every recorded token, after exporting their source assets.
subprocess.run([sys.executable,str(root/'scripts'/'build_temporal_assets.py'),'--repo',str(root)],check=True)
artifacts.extend(['temporal-energy','temporal-magnitude','temporal-left-ipr','temporal-right-ipr'])
print(json.dumps({'summary':summ,'zero_point_diagnostic':zero_diag,'assets':artifacts,'binary_bytes':sum((out/name).stat().st_size for name in state_files.values())},indent=2))
