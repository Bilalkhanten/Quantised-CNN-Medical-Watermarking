import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
INK,MUT,GRID='#1f2933','#6b7785','#e4e7eb'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK})
F=dict(psnr=40.1,rec=94.0,dice=0.77,auc=0.983)
G=[('Training and\ncoding',[('(i) No attack layer',40.2,54.6,.76,.981),('(ii) No BCH coding',40.1,0.7,.77,.983)]),
('Spatial\ntransformer',[('(iii) No spatial transformer',40.1,81.3,.75,.979)]),
('Encoder design',[('(iv) No ROI weighting',40.6,93.6,.77,.982),('(xiii) No detail-energy input D',40.0,90.9,.76,.979),('(v) Spatial-domain, no DWT',39.9,90.6,.72,.964),('(viii) Global average pooling',40.1,91.4,.77,.982)]),
('Localiser',[('(ix) No reference-pattern channel',40.2,93.3,.31,.981),('(x) Matched filter, no U-Net',40.1,94.0,.58,.983)]),
('Compression',[('(vi) INT8 instead of FP32',40.0,93.2,.76,.980),('(xi) Post-training INT8, no QAT',39.9,88.7,.71,.968),('(xii) Pruning, no fine-tuning',39.6,85.2,.70,.962)])]
rows=[];bounds=[];x=0
for name,items in G:
    s=x
    for it in items: rows.append((x,)+it); x+=1
    bounds.append((name,s,x-1)); x+=0.8   # gap between groups
xs=[r[0] for r in rows]; labels=[r[1] for r in rows]
P=[('Exact payload recovery\n(change, % points)',lambda r:r[3]-F['rec'],'{:+.1f}','#2a6fb0',(-100,12)),
   ('Tamper-localisation Dice\n(change)',lambda r:r[4]-F['dice'],'{:+.2f}','#c7701e',(-0.52,0.07)),
   ('Authentication ROC-AUC\n(change)',lambda r:r[5]-F['auc'],'{:+.3f}','#2b8a6e',(-0.024,0.004)),
   ('PSNR\n(change, dB)',lambda r:r[2]-F['psnr'],'{:+.1f}','#6b5aa6',(-0.75,0.85))]
fig,axs=plt.subplots(4,1,figsize=(15,12.8),dpi=150,sharex=True)
fig.subplots_adjust(left=.185,right=.985,top=.81,bottom=.185,hspace=.22)
for ax,(title,fn,fmt,col,lim) in zip(axs,P):
    vals=[fn(r) for r in rows]
    for i,(name,s,e) in enumerate(bounds):
        if i%2==0: ax.axvspan(xs[s]-.5 if False else rows[[r[1] for r in rows].index(G[i][1][0][0])][0]-.5, rows[[r[1] for r in rows].index(G[i][1][-1][0])][0]+.5, color='#f3f5f7',zorder=0,lw=0)
    ax.bar(xs,vals,width=.62,color=col,zorder=3)
    ax.axhline(0,color=INK,lw=1.2,zorder=4)
    ax.set_ylim(*lim); ax.set_xlim(-.7,xs[-1]+.7)
    ax.set_ylabel(title,fontsize=11.5,rotation=0,ha='right',va='center',labelpad=10)
    ax.yaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True)
    ax.tick_params(axis='y',labelsize=10,colors=MUT,length=0); ax.tick_params(axis='x',length=0)
    for s in ('top','right','left','bottom'): ax.spines[s].set_visible(False)
    big=sorted(range(len(vals)),key=lambda i:-abs(vals[i]))
    thr=max(abs(v) for v in vals)*0.18
    for i in range(len(vals)):
        v=vals[i]
        if abs(v)>=thr:
            off=5 if v>=0 else -5
            ax.annotate(fmt.format(v),(xs[i],v),xytext=(0,off),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=9.5,color=INK)
axs[-1].set_xticks(xs); axs[-1].set_xticklabels(labels,rotation=38,ha='right',fontsize=10.5,rotation_mode='anchor')
# group labels above first panel
for name,s,e in bounds:
    xa=rows[[r[1] for r in rows].index(next(it[0] for g,its in G if g==name for it in its))][0]
    xb=rows[[r[1] for r in rows].index(next(it[0] for g,its in G if g==name for it in its[-1:]))][0]
    axs[0].annotate(name,((xa+xb)/2,1.0),xycoords=('data','axes fraction'),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',fontsize=11.5,fontweight='bold',color=INK)
fig.text(.012,.970,'Effect of removing or replacing each component of the proposed model',fontsize=20,fontweight='bold',ha='left',va='center')
fig.text(.012,.942,'Change relative to the full model (PSNR 40.1 dB, exact recovery 94.0 %, Dice 0.77, AUC 0.983). Bars below zero mean the variant is worse.',fontsize=12,color=MUT,ha='left',va='center')
fig.text(.012,.908,'ILLUSTRATIVE DEMONSTRATION VALUES – NOT EXPERIMENTAL MEASUREMENTS',fontsize=12.5,fontweight='bold',color='#9b2c2c',ha='left',va='center',bbox=dict(boxstyle='round,pad=0.35',fc='#fdecec',ec='#e3a6a6',lw=1))
fig.text(.012,.012,'Variant (vii), embedding strength α × 0.7 to × 1.4, is a sweep rather than a component and is omitted here; its range is in the Table 12 values.',fontsize=10,color=MUT,ha='left')
import sys; fig.savefig(sys.argv[1] if len(sys.argv)>1 else 'Fig_ablation_components.png',dpi=150)
