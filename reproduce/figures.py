"""Scientific replot of Figs. 2--6 from exact frozen numeric artists.

Run from the repository root: python -m reproduce.figures --output outputs/figures
The archived publication PDFs retain the exact typeset artwork. This renderer
uses portable open fonts and is NOT claimed to be pixel-identical to those PDFs.
No fitting, optimization, interpolation, resampling or statistic recomputation
occurs here; original samples, masks, line connections and CI paths are read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm, BoundaryNorm, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, VPacker
from matplotlib.patches import PathPatch, Circle, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/figures'


def numbers(value):
    return np.asarray(value, dtype=float)


def line_style(record):
    if str(record.get('linestyle','-')).lower() in ('none','', ' '):
        return 'None'
    dash=record.get('dash',[0,None])
    return '-' if dash[1] is None else (dash[0],tuple(dash[1]))


def _glyph(center, color, marker, size, transform):
    """Native-vector soft highlight on the original marker shape."""
    rgb=np.asarray(to_rgb(color)); shape=MarkerStyle('o' if marker=='.' else marker)
    path=shape.get_path().transformed(shape.get_transform()).transformed(
        Affine2D().scale(size).translate(*center))
    base=PathPatch(path,facecolor=rgb*.78,edgecolor=rgb*.68,linewidth=.22,
                   transform=transform,joinstyle='round')
    artists=[base]; radius=size/2
    light=(center[0]-.27*radius,center[1]+.29*radius)
    for extent in np.linspace(1.20,.025,28):
        strength=math.exp(-.5*(extent/.46)**2)
        patch=Circle(light,radius*extent,facecolor=rgb+(1-rgb)*(.88*strength),
                     edgecolor='none',transform=transform)
        patch.set_clip_path(base);artists.append(patch)
    return artists


def soft_points(ax,x,y,color,marker,size,z):
    count=0
    for xx,yy in zip(x,y):
        if not np.isfinite([xx,yy]).all():continue
        area=DrawingArea(size,size,0,0)
        for item in _glyph((size/2,size/2),color,marker,size,area.get_transform()):area.add_artist(item)
        ax.add_artist(AnnotationBbox(area,(xx,yy),xycoords='data',box_alignment=(.5,.5),
                                     frameon=False,pad=0,annotation_clip=True,zorder=z))
        count+=1
    return count


class SoftHandler(HandlerBase):
    def create_artists(self,legend,handle,xd,yd,width,height,fontsize,transform):
        y=height/2-yd; result=[]
        if str(handle.get_linestyle()).lower() not in ('none','', ' '):
            phase,dashes=handle._unscaled_dash_pattern
            ls='-' if dashes is None else (phase,tuple(dashes))
            result.append(Line2D([-xd,width-xd],[y,y],color=handle.get_color(),
                                 lw=handle.get_linewidth(),linestyle=ls,transform=transform))
        marker=handle.get_marker()
        if str(marker).lower() not in ('none','', ' '):
            result.extend(_glyph((width/2-xd,y),handle.get_color(),marker,3.6,transform))
        return result or [Line2D([],[],transform=transform)]


def _rgba(value, index=0):
    if value is None or len(value)==0:return 'none'
    a=np.asarray(value)
    return a.tolist() if a.ndim==1 else a[index%len(a)].tolist()


def add_collection(ax,record):
    kind=record['kind']; counts={'paths':0,'mesh_cells':0,'scatter_points':0}
    if kind=='mesh':
        xy=numbers(record['coordinates']); values=np.ma.masked_invalid(numbers(record['values']))
        cmap=ListedColormap(record['colors'])
        if record['norm']=='log':norm=LogNorm(record['vmin'],record['vmax'])
        elif record['norm']=='boundary':norm=BoundaryNorm(record['boundaries'],cmap.N)
        else:norm=Normalize(record['vmin'],record['vmax'])
        item=ax.pcolormesh(xy[...,0],xy[...,1],values,cmap=cmap,norm=norm,shading='flat',
                           rasterized=True,zorder=record['z'],edgecolors='none',alpha=record['alpha'])
        assert np.ma.allequal(item.get_array(),values)
        counts['mesh_cells']=values.size
    elif kind=='scatter':
        xy=numbers(record['offsets']); marker_path=record['marker_paths'][0]
        marker=MplPath(numbers(marker_path['vertices']),None if marker_path['codes'] is None else np.asarray(marker_path['codes'],np.uint8))
        color=_rgba(record['face'])
        if record.get('soft_marker'):
            counts['scatter_points']=soft_points(ax,xy[:,0],xy[:,1],color,record['marker'],
                                                 math.sqrt(record['sizes'][0]),record['z']+1)
        else:
            ax.scatter(xy[:,0],xy[:,1],s=record['sizes'],marker=marker,facecolors=color,
                       edgecolors=_rgba(record['edge']),linewidths=record['widths'],
                       alpha=record['alpha'],zorder=record['z'],clip_on=False)
            counts['scatter_points']=len(xy)
    else:
        for i,path in enumerate(record['paths']):
            vertices=numbers(path['vertices']); codes=path['codes']
            style=record.get('dashes',[[0,None]])
            dash=style[i%len(style)] if style else [0,None]
            ls='-' if dash[1] is None else (dash[0],tuple(dash[1]))
            patch=PathPatch(MplPath(vertices,None if codes is None else np.asarray(codes,np.uint8)),
                            transform=ax.transData,facecolor=_rgba(record['face'],i),
                            edgecolor=_rgba(record['edge'],i),
                            linewidth=record['widths'][i%len(record['widths'])] if record['widths'] else 0,
                            linestyle=ls,alpha=record['alpha'],zorder=record['z'])
            ax.add_patch(patch);counts['paths']+=1
            assert np.array_equal(patch.get_path().vertices,vertices,equal_nan=True)
    return counts


def add_legend(ax,record):
    handles=[]
    for row in record['handles']:
        handles.append(Line2D([],[],color=row['color'],lw=row['width'],linestyle=line_style(row),
                              marker=row['marker'],markersize=3.6,label=row['label']))
    kwargs=dict(handles=handles,loc=record['loc'],bbox_to_anchor=record['anchor'],
                ncol=record['ncol'],fontsize=record['font_size'],frameon=record['frame'],
                borderpad=.25,labelspacing=.15,handlelength=1.3,handletextpad=.35,
                columnspacing=.7,borderaxespad=0,handler_map={Line2D:SoftHandler()})
    title=record['title']
    if title:kwargs.update(title=title,title_fontsize=record['font_size'])
    legend=ax.legend(**kwargs)
    if record['compact_rate_key']:
        # Keep lambda_R as a side title, with the CDF/rANS or reference row
        # inside the same corner frame. Numeric labels are not new series.
        columns=legend._legend_handle_box.get_children()
        if len(handles)==6:
            upper_columns=[VPacker(pad=0,sep=c.sep,align='baseline',children=c.get_children()[:2]) for c in columns]
            upper=HPacker(pad=0,sep=.7*6.9,align='baseline',children=upper_columns)
            upper=HPacker(pad=0,sep=3.2,align='center',children=[legend._legend_title_box,upper])
            lower=HPacker(pad=0,sep=.7*6.9,align='baseline',children=[c.get_children()[2] for c in columns])
            legend._legend_box=VPacker(pad=0,sep=2.,align='center',children=[upper,lower])
        else:
            legend._legend_box=HPacker(pad=0,sep=3.2,align='center',children=[legend._legend_title_box,legend._legend_handle_box])
        legend._legend_box.set_figure(ax.figure);legend._legend_box.axes=ax
        legend._legend_box.set_offset(legend._findoffset)
    legend.get_frame().set_facecolor('white');legend.get_frame().set_alpha(1)
    legend.get_frame().set_edgecolor('#C0C0C0');legend.get_frame().set_linewidth(.45)
    ax.add_artist(legend);ax.legend_=None


def replot(number, output):
    scene=json.loads((SOURCE/f'fig{number}_scene.json').read_text(encoding='utf-8'))
    assert scene['schema']=='finite-mode-frozen-scientific-scene-v1'
    size=list(scene['size_inches'])
    if number==3:size[0]=4.0  # open-font tick widths need slightly more room
    fig=plt.figure(figsize=size);fig.set_dpi(100)
    counts={'lines':0,'soft_observations':0,'scatter_points':0,'paths':0,'mesh_cells':0}
    for row in scene['axes']:
        ax=fig.add_axes(row['position'])
        ax.set_xscale(row['xscale']);ax.set_yscale(row['yscale'])
        for col in row['collections']:
            for key,value in add_collection(ax,col).items():counts[key]+=value
        for patch in row['patches']:
            ax.add_patch(PathPatch(MplPath(numbers(patch['vertices']),patch['codes']),facecolor=patch['face'],
                                   edgecolor=patch['edge'],linewidth=patch['width'],alpha=patch['alpha'],zorder=patch['z']))
        for line in row['lines']:
            if not line['visible']:continue
            x,y=numbers(line['x']),numbers(line['y'])
            marker=line['marker']
            item,=ax.plot(x,y,color=line['color'],lw=line['width'],linestyle=line_style(line),
                          marker='None' if line['soft_marker'] else marker,markersize=line['marker_size'],
                          alpha=line['alpha'],zorder=line['z'])
            assert np.array_equal(item.get_xdata(),x,equal_nan=True) and np.array_equal(item.get_ydata(),y,equal_nan=True)
            if line['soft_marker']:
                counts['soft_observations']+=soft_points(ax,x,y,line['color'],marker,line['marker_size'],max(4,line['z']+1))
            counts['lines']+=1
        ax.set_xlim(row['xlim']);ax.set_ylim(row['ylim'])
        ax.set_xticks(row['xticks'],row['xticklabels']);ax.set_yticks(row['yticks'],row['yticklabels'])
        ax.set_xlabel(row['xlabel'],fontsize=8.3,labelpad=3);ax.set_ylabel(row['ylabel'],fontsize=8.3,labelpad=4)
        ax.set_title(row['title'],fontsize=8.3,pad=2)
        if row['grid']:ax.grid(True,color='#D8D8D8',lw=.45,ls='--',zorder=-1)
        else:ax.grid(False,which='both')
        ax.tick_params(direction='in',labelsize=8,width=.7,length=2.5)
        if row['ytick_side']=='right':ax.yaxis.tick_right()
        for name,spine in ax.spines.items():
            old=row['spines'].get(name,row['spines'].get('outline',{'visible':False,'width':.7}))
            spine.set_visible(old['visible']);spine.set_linewidth(old['width']);spine.set_color('#555555')
        if 'outline' in row['spines'] and row['spines']['outline']['visible']:
            ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor='none',edgecolor='#555555',
                                   linewidth=row['spines']['outline']['width'],clip_on=False,zorder=10))
        for text in row['texts']:
            kwargs={'bbox':{'facecolor':'white','edgecolor':'none','pad':.2}} if text['bbox'] else {}
            ax.text(*text['xy_axes'],text['text'],transform=ax.transAxes,fontsize=text['size'],color=text['color'],
                    ha=text['ha'],va=text['va'],rotation=text['rotation'],zorder=text['z'],clip_on=False,**kwargs)
        for key in row['legends']:add_legend(ax,key)
        for top in row['top_axes']:
            secondary=ax.secondary_xaxis('top')
            secondary.set_xticks(top['ticks_parent'],top['labels']);secondary.set_xlabel(top['xlabel'],fontsize=8.3,labelpad=2)
            secondary.tick_params(direction='in',labelsize=8,width=.6,length=2)
    output.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):
        fig.savefig(output/f'fig{number}_replot.{extension}',dpi=300,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    return {'figure':number,'status':'PASS','numeric_artist_counts':counts,
            'pixel_identity_claimed':False,'data_modified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
    parser.add_argument('--figures',type=int,nargs='+',choices=range(2,7),default=list(range(2,7)))
    args=parser.parse_args(argv)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','font.size':8,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
    result=[replot(number,args.output) for number in args.figures]
    (args.output/'replot_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','figures':len(result),'output':str(args.output)}))


if __name__=='__main__':main()
