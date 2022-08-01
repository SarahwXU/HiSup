import os
import cv2
import numpy as np
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as Patches

from descartes import PolygonPatch


# set color maps for visulization
# colormap = mpl.cm.Paired.colors
# num_color = len(colormap)

colormap = mpl.cm.Paired.colors
colormap = (
    (0.6509803921568628, 0.807843137254902, 0.8901960784313725), 
    (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
    (0.984313725490196, 0.6039215686274509, 0.6), 
    (0.8901960784313725, 0.10196078431372549, 0.10980392156862745), 
    (0.9921568627450981, 0.7490196078431373, 0.43529411764705883), 
    (1.0, 0.4980392156862745, 0.0), 
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098), 
    (0.41568627450980394, 0.23921568627450981, 0.6039215686274509), 
    (1.0, 1.0, 0.6), 
    (0.6941176470588235, 0.34901960784313724, 0.1568627450980392))

num_color = len(colormap)


def show_polygons(image, polys):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.fill(polygon[:,0], polygon[:, 1], color=color, alpha=0.3)
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    plt.show()

def save_viz(image, polys, save_path, filename):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    impath = osp.join(save_path, 'viz', 'crowdai_val_small', filename)
    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def viz_inria(image, polygons, output_dir, file_name, alpha=0.5, linewidth=12, markersize=45):
    plt.rcParams['figure.figsize'] = (500,500)
    plt.rcParams['figure.dpi'] = 10
    plt.axis('off')
    plt.imshow(image)
    for n, poly in enumerate(polygons):
        poly_color = colormap[n%num_color]
        if poly.type == 'MultiPolygon':
            for p in poly:
                patch = PolygonPatch(p.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
                plt.gca().add_patch(Patches.Polygon(p.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                juncs = np.array(p.exterior.coords[:-1])
                plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
                if len(p.interiors) != 0:
                    for inter in p.interiors:
                        plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                        juncs = np.array(inter.coords[:-1])
                        plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
        else:
            try:
                patch = PolygonPatch(poly.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
            except TypeError:
                plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=True, ec=poly_color, fc=poly_color, linewidth=linewidth, alpha=alpha))
            plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
            juncs = np.array(poly.exterior.coords[:-1])
            plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
            if len(poly.interiors) != 0:
                for inter in poly.interiors:
                    plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                    juncs = np.array(inter.coords[:-1])
                    plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
    
    # save_filename = os.path.join(output_dir, 'inria_viz', file_name[:-4] + '.svg')
    # plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
    # plt.clf()
    plt.show()


def draw_predictions_with_mask_inria(img, junctions, polys_ids, save_dir, filename):
    plt.axis('off')
    plt.imshow(img)

    instance_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for h, idx in enumerate(contours):
            poly = junctions[idx]
            if h == 0:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=color, thickness=-1)
            else:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=0, thickness=-1)

            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=2))
    
    alpha_map = np.bitwise_or(instance_mask[:,:,0:1].astype(bool), 
                              instance_mask[:,:,1:2].astype(bool), 
                              instance_mask[:,:,2:3].astype(bool)).astype(np.float32)
    instance_mask = np.concatenate((instance_mask, alpha_map), axis=-1)
    plt.imshow(instance_mask, alpha=0.3)
    plt.show()


def draw_predictions_inria(img, junctions, polys_ids):
    plt.axis('off')

    plt.imshow(img)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for idx in contours:
            poly = junctions[idx]
            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=1.5))
            plt.plot(poly[:,0], poly[:,1], color=color, marker='.')
    plt.show()




