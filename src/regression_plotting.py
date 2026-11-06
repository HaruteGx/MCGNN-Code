import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator

DEFAULT_DPI = 600
CLASS_COLORS = ('#267A9E', '#D38A31', '#7966A3')
CMAP = LinearSegmentedColormap.from_list('class_blue', ['#F6F9FB', '#B9D7E5', '#1B5D7B'])
STYLE = {'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':10,
         'axes.titlesize':11, 'axes.titleweight':'bold', 'axes.linewidth':.8,
         'xtick.labelsize':9, 'ytick.labelsize':9, 'svg.fonttype':'none',
         'savefig.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False}


def dump_json(path, values):
    Path(path).write_text(json.dumps(values,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def save_figure(fig, directory, stem, dpi=DEFAULT_DPI):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(directory/f'{stem}.{extension}',dpi=dpi,bbox_inches='tight',pad_inches=.18)
    fig.savefig(directory/f'{stem}_preview.png',dpi=125,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def validate_boundaries(boundaries):
    cuts=np.asarray(boundaries,dtype=float)
    if cuts.shape!=(2,) or not np.isfinite(cuts).all() or cuts[0]>=cuts[1]:
        raise ValueError('Three-class plotting requires two finite increasing boundaries.')
    return cuts


def interval_metrics(truth, predicted, boundaries):
    cuts=validate_boundaries(boundaries)
    truth,predicted=np.asarray(truth,dtype=float),np.asarray(predicted,dtype=float)
    if truth.ndim!=1 or truth.shape!=predicted.shape or len(truth)==0:
        raise ValueError('Expected equally sized nonempty 1-D paired G arrays.')
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError('Paired G values must be finite.')
    a,b=np.searchsorted(cuts,truth,side='right'),np.searchsorted(cuts,predicted,side='right')
    cm=np.bincount(3*a+b,minlength=9).reshape(3,3)
    counts,pc=cm.sum(1),cm.sum(0)
    recall=np.divide(cm.diagonal(),counts,out=np.zeros(3),where=counts>0)
    precision=np.divide(cm.diagonal(),pc,out=np.zeros(3),where=pc>0)
    f1=np.divide(2*cm.diagonal(),counts+pc,out=np.zeros(3),where=counts+pc>0)
    error=predicted-truth
    variance=np.sum((truth-truth.mean())**2)
    return {'n':len(truth),'correct':int(cm.trace()),'accuracy':float(cm.trace()/len(truth)),
            'balanced_accuracy':float(recall[counts>0].mean()),'macro_f1':float(f1.mean()),
            'counts':counts.tolist(),'predicted_counts':pc.tolist(),'confusion':cm.tolist(),
            'recalls':[float(recall[i]) if counts[i] else None for i in range(3)],
            'precisions':[float(precision[i]) if pc[i] else None for i in range(3)],
            'f1_per_class':f1.tolist(),'g_mae':float(np.abs(error).mean()),
            'g_rmse':float(np.sqrt(np.mean(error**2))),
            'g_r2':float(1-np.sum(error**2)/variance) if variance>0 else None,
            'boundaries':cuts.tolist()}


def prepare_predictions(frame,boundaries):
    cuts=validate_boundaries(boundaries)
    predicted=pd.to_numeric(frame['Predicted G (kcal/mol)'],errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(predicted)
    if 'status' in frame:
        valid &= frame.status.eq('OK').to_numpy()
    truth_column=next((c for c in ('True G (kcal/mol)','G (kcal/mol)') if c in frame),None)
    truth=(pd.to_numeric(frame[truth_column],errors='coerce').to_numpy(dtype=float)
           if truth_column else np.full(len(frame),np.nan))
    paired=valid & np.isfinite(truth)

    for col,values,mask in [('Predicted Class',predicted,valid),('True Class',truth,paired)]:
        if col in frame:
            saved=pd.to_numeric(frame[col],errors='coerce').to_numpy(dtype=float,na_value=np.nan)
            if not np.isfinite(saved[mask]).all() or not np.array_equal(
                    saved[mask],np.searchsorted(cuts,values[mask],side='right')):
                raise ValueError(f'{col} is inconsistent with these boundaries; use the configured regression boundaries.')
    coverage={'total_rows':len(frame),'valid_predictions':int(valid.sum()),
              'paired_rows':int(paired.sum()),'unscored_predictions':int((valid & ~paired).sum()),
              'failed_or_nonfinite_rows':int((~valid).sum())}
    return truth,predicted,valid,paired,coverage


def draw_confusion(ax,metrics,title='Confusion matrix'):
    cm=np.asarray(metrics['confusion'])
    counts=cm.sum(1)
    norm=np.divide(cm,counts[:,None],out=np.zeros((3,3)),where=counts[:,None]>0)
    ax.imshow(norm,cmap=CMAP,vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            percent=f'{norm[i,j]:.1%}' if counts[i] else 'n.a.'
            ax.text(j,i,f'{cm[i,j]}\n{percent}',ha='center',va='center',fontsize=11,
                    color='white' if norm[i,j]>.65 else '#263744')
    ax.set_xticks(range(3),['Low G','Mid G','High G'])
    ax.set_yticks(range(3),['Low G','Mid G','High G'])
    ax.set(xlabel='Predicted class',ylabel='Experimental G class',title=title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_parity(ax,truth,predicted,cuts,metrics):
    a,b=np.searchsorted(cuts,truth,side='right'),np.searchsorted(cuts,predicted,side='right')
    same=a==b
    lo=min(truth.min(),predicted.min(),cuts[0])-.6
    hi=max(truth.max(),predicted.max(),cuts[1])+.6
    ax.plot([lo,hi],[lo,hi],'--',color='#555555',lw=1)
    for cut in cuts:
        ax.axvline(cut,color='#888888',ls=':',lw=.9)
        ax.axhline(cut,color='#888888',ls=':',lw=.9)
    ax.scatter(truth[same],predicted[same],s=19,color=CLASS_COLORS[0],alpha=.65,label='Class agrees')
    ax.scatter(truth[~same],predicted[~same],s=23,color='#C15A2C',marker='x',lw=.8,label='Class differs')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r'Experimental $\Delta G$ (kcal/mol)',
           ylabel=r'Predicted $\Delta G$ (kcal/mol)',title='Continuous G output')
    ax.set_aspect('equal',adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    r2=f'{metrics["g_r2"]:.3f}' if metrics['g_r2'] is not None else 'n.a.'
    ax.text(.96,.03,f'$R^2$ = {r2}\nMAE = {metrics["g_mae"]:.3f} kcal/mol',transform=ax.transAxes,
            ha='right',va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
    ax.legend(loc='upper left',frameon=False,fontsize=8)


def prediction_report(frame,directory,*,boundaries,title,role_note='',dpi=DEFAULT_DPI,stem='classification_report'):
    cuts=validate_boundaries(boundaries)
    truth,predicted,valid,paired,coverage=prepare_predictions(frame,cuts)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    metrics=interval_metrics(truth[paired],predicted[paired],cuts) if paired.any() else None
    report={'coverage':coverage,'metrics':metrics,'boundaries':cuts.tolist(),'evaluation_role':role_note}
    with plt.rc_context(STYLE):
        if metrics is not None:
            fig,axs=plt.subplots(2,2,figsize=(10.2,9.5))
            fig.subplots_adjust(left=.10,right=.96,top=.85,bottom=.17,hspace=.53,wspace=.40)
            draw_parity(axs[0,0],truth[paired],predicted[paired],cuts,metrics)
            draw_confusion(axs[0,1],metrics)
            x=np.arange(3)
            for offset,key,label,color in [(-.24,'precisions','Precision',CLASS_COLORS[0]),
                     (0.,'recalls','Recall',CLASS_COLORS[1]),(.24,'f1_per_class','F1',CLASS_COLORS[2])]:
                vals=np.array([np.nan if v is None else v for v in metrics[key]])
                axs[1,0].bar(x+offset,vals,width=.22,label=label,color=color)
            axs[1,0].set(ylim=(0,1.05),title='Performance of each class',ylabel='Score')
            axs[1,0].set_xticks(x,['Low G','Mid G','High G'])
            axs[1,0].yaxis.set_major_formatter(PercentFormatter(1))
            axs[1,0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3)
            axs[1,1].bar(x-.18,metrics['counts'],width=.34,color=CLASS_COLORS[0],label='Experimental')
            axs[1,1].bar(x+.18,metrics['predicted_counts'],width=.34,color=CLASS_COLORS[1],label='Predicted')
            axs[1,1].set_xticks(x,['Low G','Mid G','High G'])
            axs[1,1].set(title='Class counts (paired records)',ylabel='Number of records')
            axs[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
            axs[1,1].legend(frameon=False,fontsize=8)
            summary=(f'Accuracy {metrics["accuracy"]:.2%} ({metrics["correct"]}/{metrics["n"]})'
                     f'    Balanced accuracy {metrics["balanced_accuracy"]:.2%}    Macro F1 {metrics["macro_f1"]:.2%}')
        else:
            fig,axs=plt.subplots(1,2,figsize=(10.2,4.9))
            fig.subplots_adjust(left=.10,right=.96,top=.74,bottom=.25,wspace=.32)
            values=predicted[valid]
            if len(values):
                bins=np.histogram_bin_edges(values,bins=min(25,max(3,int(np.sqrt(len(values))))))
                for c in range(3):
                    members=values[np.searchsorted(cuts,values,side='right')==c]
                    axs[0].hist(members,bins=bins,alpha=.7,color=CLASS_COLORS[c],label=f'Class {c}')
                counts=np.bincount(np.searchsorted(cuts,values,side='right'),minlength=3)
            else:
                counts=np.zeros(3,dtype=int)
                axs[0].text(.5,.5,'No valid predictions',ha='center',transform=axs[0].transAxes)
            for cut in cuts:
                axs[0].axvline(cut,color='#555555',ls=':',lw=1)
            axs[0].set(xlabel=r'Predicted $\Delta G$ (kcal/mol)',ylabel='Number of records',title='Predicted G distribution')
            axs[1].bar(range(3),counts,color=CLASS_COLORS)
            axs[1].set_xticks(range(3),['Low G','Mid G','High G'])
            axs[1].set(title='Predicted class counts',ylabel='Number of records')
            summary='No paired experimental G: classification accuracy is unavailable'
        fig.suptitle(title,fontsize=15,fontweight='bold',y=.985)
        fig.text(.53,.93,f'Boundaries: {cuts[0]:g}, {cuts[1]:g} kcal/mol; equality enters the higher class',ha='center',fontsize=10)
        fig.text(.53,.89,summary,ha='center',fontsize=10)
        detail=('Matrix cells: count / row percentage (color scale 0–100%).' if metrics is not None
                else 'Distributions use valid predictions only.')
        fig.text(.10,.035,f'{role_note}\n'
                 f'Paired {coverage["paired_rows"]}/{coverage["total_rows"]}; predictions without truth {coverage["unscored_predictions"]}; '
                 f'failed/nonfinite {coverage["failed_or_nonfinite_rows"]}. '
                 f'{detail}',fontsize=8.3,linespacing=1.5)
        save_figure(fig,directory,stem,dpi)
    dump_json(directory/f'{stem}.json',report)
    return report
