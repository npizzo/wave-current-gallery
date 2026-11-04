import numpy as np, matplotlib as mpl, sys, os
mpl.use('Agg'); import matplotlib.pyplot as plt
deep,none,tmp,i0,i1=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
plt.rcParams.update({'font.size':11,'font.family':'serif','mathtext.fontset':'cm'})
A=[np.load(none),np.load(deep)]; names=['3D, no waves','3D CL, deep Stokes drift']
n=min(len(a['t']) for a in A); t=A[0]['t'][:n]; assert np.allclose(t,A[1]['t'][:n]); i1=min(i1,n)
z=A[0]['z']; vmax=2.5; T=200.
fig=plt.figure(figsize=(16,4.8),dpi=120)
gs=fig.add_gridspec(1,7,width_ratios=[1,0.9,0.14,0.06,1,0.9,0.05],wspace=0.1,left=0.045,right=0.95,bottom=0.15,top=0.83)
ims=[];hov=[];cur=[]
for k,(a,(cu,ch)) in enumerate(zip(A,[(0,1),(4,5)])):
    ax=fig.add_subplot(gs[0,cu]); im=ax.imshow(a['uxz'][0].astype(float).T,origin='lower',extent=[0,1,-1,0],cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='equal',interpolation='nearest')
    ax.set_xticks([0,0.5,1]); ax.set_xticklabels(['0','0.5','1']); ax.set_xlabel('$x$'); ax.set_ylabel('$z$'); ax.set_title(f'{names[k]}: slice $u(x,y{{=}}0,z)$',fontsize=11); ims.append(im)
    ax2=fig.add_subplot(gs[0,ch]); H=np.full((n,len(z)),np.nan); H[:i0]=a['UL'][:i0]
    hm=ax2.pcolormesh(t,z,H.T,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='nearest'); hov.append((hm,H,a))
    ax2.set_xlim(0,T); ax2.set_ylim(-1,0); ax2.set_yticklabels([]); ax2.set_xlabel('$t/T_e$')
    ax2.set_title(r'mean $U_L(z,t)=\langle u\rangle_{x,y}$',fontsize=11); cur.append(ax2.axvline(0,color='k',lw=1))
cax=fig.add_subplot(gs[0,6]); cb=fig.colorbar(ims[0],cax=cax,extend='both'); cb.set_label('velocity $/\\,u_0$')
ttl=fig.suptitle('',fontsize=13,y=0.975)
os.makedirs(tmp,exist_ok=True)
for i in range(i0,i1):
    for k,a in enumerate(A):
        ims[k].set_data(a['uxz'][i].astype(float).T); hm,H,_=hov[k]; H[i]=a['UL'][i]; hm.set_array(H.T.ravel()); cur[k].set_xdata([t[i],t[i]])
    ttl.set_text(f'Deep-water Vallis ring in 3D ($r_b=1$, 3D seed $10^{{-2}}$ of ring energy):   $t={t[i]:5.1f}\\,T_e$')
    fig.savefig(f'{tmp}/f{i:04d}.png',dpi=120)
print('frames',i0,i1,'of',n)
