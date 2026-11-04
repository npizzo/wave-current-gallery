import numpy as np, matplotlib as mpl, sys, os, glob, json
mpl.use('Agg'); import matplotlib.pyplot as plt
D=os.path.expanduser('~/drfig'); OUT=sys.argv[1]
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
B,O='#2a6fb0','#d9822b'
def ts(r): return np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True)
fig,axs=plt.subplots(2,2,figsize=(7.2,5.6),gridspec_kw=dict(hspace=0.38,wspace=0.28,left=0.09,right=0.98,bottom=0.09,top=0.93))
for c,rb in enumerate(['1','0.5']):
    ax=axs[0,c]
    fam=[(f'd3-deep-rb{rb}-*',B,'-','3D CL, deep waves'),(f's2d-deep-rb{rb}-*',B,'--','2D, deep waves'),(f'd3-none-rb{rb}-*',O,'-','3D, no waves'),(f's2d-none-rb{rb}-*',O,'--','2D, no waves')]
    for pat,col,ls,lab in fam:
        for k,d in enumerate(sorted(glob.glob(f'{D}/{pat}'))):
            t=ts(os.path.basename(d)); ax.plot(t['t_Te'],t['KEjet']/t['KE'],color=col,ls=ls,lw=1.0 if ls=='-' else 0.9,alpha=0.85,label=lab if k==0 else None)
    ax.set_title(f'jet fraction $\\frac{{1}}{{2}}\\langle U_L^2\\rangle/\\mathrm{{KE}}$,  $r_b={rb}$'); ax.set_xlabel('$t/T_e$'); ax.set_xlim(0,200); ax.set_ylim(0,0.5)
    ax.grid(color='0.9',lw=0.5)
    if c==0: ax.legend(fontsize=7.5,frameon=False,loc='upper left')
# sQ diagnostics over all deep 3D runs
ax=axs[1,0]; ax2=axs[1,1]
for d in sorted(glob.glob(f'{D}/d3-deep-*')):
    r=os.path.basename(d); nz=256; z=np.loadtxt(f'{d}/zc.txt'); p=np.loadtxt(f'{d}/profiles.csv',delimiter=',',skiprows=1)
    t=p[:,0]; UL=p[:,1:1+nz]; E3=p[:,1+nz:1+2*nz]; E2=p[:,1+2*nz:]; usz=0.25*np.exp(8*z)
    fn=[];fp=[];dn=[]
    for i in range(len(t)):
        neg=(1-np.gradient(UL[i],z)/usz)<0; dn.append(neg.mean())
        fn.append(E3[i][neg].sum()/(E3[i][neg]+E2[i][neg]).sum() if neg.sum()>2 else np.nan)
        fp.append(E3[i][~neg].sum()/(E3[i][~neg]+E2[i][~neg]).sum())
    fn=np.array(fn); fp=np.array(fp)
    lab=r=='d3-deep-rb1-e1e-2-s1'
    ax.plot(t,fn,'.',ms=1.6,color='#7a3e9d',alpha=0.6,label='levels with $sQ<0$' if lab else None)
    ax.plot(t,fp,'-',lw=0.8,color='0.45',alpha=0.9,label='levels with $sQ>0$' if lab else None)
    ax2.plot(t,dn,'-',lw=0.8,color='0.35',alpha=0.8)
ax.set_yscale('log'); ax.set_ylim(1e-3,1.2); ax.set_xlim(0,200); ax.set_xlabel('$t/T_e$'); ax.set_title('y-dependent energy fraction (12 deep 3D runs)',fontsize=9); ax.legend(fontsize=7.5,frameon=False,loc='lower left',markerscale=5); ax.grid(color='0.9',lw=0.5)
ax2.set_xlim(0,200); ax2.set_ylim(0,0.25); ax2.set_xlabel('$t/T_e$'); ax2.set_title('fraction of depth with $sQ<0$',fontsize=9); ax2.text(0.97,0.95,'$sQ=\\partial_z u^s(\\partial_z u^s-\\partial_z U_L)$',transform=ax2.transAxes,ha='right',va='top',fontsize=8); ax2.grid(color='0.9',lw=0.5)
fig.savefig(f'{OUT}/diagnostics.png',dpi=200); plt.close(fig)
