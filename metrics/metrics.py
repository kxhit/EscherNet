import numpy as np
from scipy.spatial import cKDTree as KDTree
import mesh2sdf
import open3d

def chamfer(gt_points, rec_points):
    # one direction
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(rec_points)
    gen_to_gt_chamfer = np.mean(two_distances)

    return (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.

# compute volume iou
def compute_iou(mesh_pr, mesh_gt):
    # trimesh to open3d
    mesh_gt_o3d = open3d.geometry.TriangleMesh()
    mesh_gt_o3d.vertices = open3d.utility.Vector3dVector(mesh_gt.vertices)
    mesh_gt_o3d.triangles = open3d.utility.Vector3iVector(mesh_gt.faces)
    mesh_rec_o3d = open3d.geometry.TriangleMesh()
    mesh_rec_o3d.vertices = open3d.utility.Vector3dVector(mesh_pr.vertices)
    mesh_rec_o3d.triangles = open3d.utility.Vector3iVector(mesh_pr.faces)

    size = 64
    sdf_pr = mesh2sdf.compute(mesh_rec_o3d.vertices, mesh_rec_o3d.triangles, size, fix=False, return_mesh=False)
    sdf_gt = mesh2sdf.compute(mesh_gt_o3d.vertices, mesh_gt_o3d.triangles, size, fix=False, return_mesh=False)
    vol_pr = sdf_pr<0
    vol_gt = sdf_gt<0
    iou = np.sum(vol_pr & vol_gt)/np.sum(vol_gt | vol_pr)
    return iou