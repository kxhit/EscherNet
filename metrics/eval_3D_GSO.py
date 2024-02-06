import os
import numpy as np
import trimesh
import open3d as o3d
from metrics import chamfer, compute_iou
from tqdm import tqdm
# seed
np.random.seed(0)

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    z = np.sqrt(xy + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])

def get_pose(target_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    return theta_target, azimuth_target, z_target

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst

def normalize_mesh(vertices):
    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    scale = 1 / np.max(max_pt - min_pt)
    vertices = vertices * scale

    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    center = (max_pt + min_pt) / 2
    vertices = vertices - center[None, :]
    return vertices

def capture_screenshots(mesh_rec_o3d, cam_param, render_param, img_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, visible=False)
    vis.add_geometry(mesh_rec_o3d)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(cam_param)
    ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
    vis.get_render_option().load_from_json(render_param)  # rgb
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_name, do_render=True)
    vis.destroy_window()
    del vis
    del ctr


def vis_3D_rec(GT_DIR, REC_DIR, method_name):
    N = 4096
    # get all folders
    obj_names = [f for f in os.listdir(GT_DIR) if os.path.isdir(os.path.join(GT_DIR, f))]

    CDs = []
    IoUs = []
    for obj_name in tqdm(obj_names):
        print(obj_name)
        gt_meshfile = os.path.join(GT_DIR, obj_name, "meshes", "model.obj")
        if "ours" in REC_DIR:
            condition_pose = np.load(os.path.join(GT_DIR, obj_name, "render_sync_36_single/model/000.npy"))
        else:
            condition_pose = np.load(os.path.join(GT_DIR, obj_name, "render_mvs_25/model/000.npy"))
            condition_pose = np.concatenate([condition_pose, np.array([[0, 0, 0, 1]])], axis=0)
        theta, azimu, radius = get_pose(condition_pose[:3, :])

        if "PointE" in REC_DIR:
            rec_pcfile = os.path.join(REC_DIR, obj_name, "pc.ply")
        if "RealFusion" in REC_DIR:
            rec_meshfile = os.path.join(REC_DIR, obj_name, "mesh/mesh.obj")
        elif "dreamgaussian" in REC_DIR:
            rec_meshfile = os.path.join(REC_DIR, obj_name+".obj")
        elif "Wonder3D" in REC_DIR:
            rec_meshfile = os.path.join(REC_DIR, "mesh-ortho-"+obj_name, "save/it3000-mc192.obj")
        else:
            rec_meshfile = os.path.join(REC_DIR, obj_name, "mesh.ply")



        mesh_gt = trimesh.load(gt_meshfile)
        mesh_gt_o3d = o3d.io.read_triangle_mesh(gt_meshfile, True)

        # trimesh load point cloud
        if "PointE" in REC_DIR:
            pc_rec = trimesh.load(rec_pcfile)

        if method_name == "GT":
            mesh_rec = trimesh.load(gt_meshfile)
            mesh_rec_o3d = o3d.io.read_triangle_mesh(gt_meshfile, True)
        else:
            mesh_rec = trimesh.load(rec_meshfile)
            mesh_rec_o3d = o3d.io.read_triangle_mesh(rec_meshfile, True)

        # normalize
        mesh_gt.vertices = normalize_mesh(mesh_gt.vertices)
        vertices_gt = np.asarray(mesh_gt_o3d.vertices)
        vertices_gt = normalize_mesh(vertices_gt)
        mesh_gt_o3d.vertices = o3d.utility.Vector3dVector(vertices_gt)


        if "PointE" in REC_DIR:
            pc_rec.vertices = normalize_mesh(pc_rec.vertices)

        # normalize
        mesh_rec.vertices = normalize_mesh(mesh_rec.vertices)
        vertices_rec = np.asarray(mesh_rec_o3d.vertices)
        vertices_rec = normalize_mesh(vertices_rec)
        mesh_rec_o3d.vertices = o3d.utility.Vector3dVector(vertices_rec)


        if "RealFusion" in REC_DIR or "Wonder3D_ours" in REC_DIR or "SyncDreamer" in REC_DIR:
            mesh_rec.vertices = trimesh.transformations.rotation_matrix(azimu[0], [0, 0, 1])[:3, :3].dot(
                mesh_rec.vertices.T).T
            # o3d
            R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([0., 0., azimu[0]]))
            mesh_rec_o3d.rotate(R, center=(0, 0, 0))
        elif "dreamgaussian" in REC_DIR:
            mesh_rec.vertices = trimesh.transformations.rotation_matrix(azimu[0]+np.pi/2, [0, 1, 0])[:3, :3].dot(
                mesh_rec.vertices.T).T
            # rotate 90 along x
            mesh_rec.vertices = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])[:3, :3].dot(
                mesh_rec.vertices.T).T
            # o3d
            R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([0., azimu[0]+np.pi/2, 0.]))
            mesh_rec_o3d.rotate(R, center=(0, 0, 0))
            R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([np.pi/2, 0., 0.]))
            mesh_rec_o3d.rotate(R, center=(0, 0, 0))
        elif "one2345" in REC_DIR:
            # rotate along z axis by azimu degree
            # mesh_rec.apply_transform(trimesh.transformations.rotation_matrix(-azimu, [0, 0, 1]))
            azimu = np.rad2deg(azimu[0])
            azimu += 60 # https://github.com/One-2-3-45/One-2-3-45/issues/26
            # print("azimu", azimu)
            mesh_rec.vertices = trimesh.transformations.rotation_matrix(np.radians(azimu), [0, 0, 1])[:3, :3].dot(mesh_rec.vertices.T).T
            # # scale again
            # mesh_rec, rec_center, rec_scale = normalize_mesh(mesh_rec)
            # o3d
            R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([0., 0., np.radians(azimu)]))
            mesh_rec_o3d.rotate(R, center=(0, 0, 0))
            # # scale again
            # mesh_rec_o3d = mesh_rec_o3d.translate(-rec_center)
            # mesh_rec_o3d = mesh_rec_o3d.scale(1 / rec_scale, [0, 0, 0])
        elif "PointE" in REC_DIR or "ShapeE" in REC_DIR:
            # sample points from rec_pc
            if "PointE" in REC_DIR:
                rec_pc_tri = pc_rec
                rec_pc_tri.vertices = rec_pc_tri.vertices[np.random.choice(np.arange(len(pc_rec.vertices)), N)]
            else:
                rec_pc = trimesh.sample.sample_surface(mesh_rec, N)
                rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

            gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
            gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
            # loop over all flips and 90 degrees rotations of rec_pc, pick the one with the smallest chamfer distance
            chamfer_dist_min = np.inf
            opt_axis = None
            opt_angle = None
            for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                for angle in [0, 90, 180, 270]:
                    tmp_rec_pc_tri = rec_pc_tri.copy()
                    tmp_rec_pc_tri.vertices = trimesh.transformations.rotation_matrix(np.radians(angle), axis)[:3, :3].dot(tmp_rec_pc_tri.vertices.T).T
                    tmp_mesh_rec = mesh_rec.copy()
                    tmp_mesh_rec.vertices = trimesh.transformations.rotation_matrix(np.radians(angle), axis)[:3, :3].dot(tmp_mesh_rec.vertices.T).T
                    # compute chamfer distance
                    chamfer_dist = chamfer(gt_pc_tri.vertices, tmp_rec_pc_tri.vertices)
                    if chamfer_dist < chamfer_dist_min:
                        chamfer_dist_min = chamfer_dist
                        opt_axis = axis
                        opt_angle = angle

            chamfer_dist = chamfer_dist_min

            mesh_rec.vertices = trimesh.transformations.rotation_matrix(np.radians(opt_angle), opt_axis)[:3, :3].dot(mesh_rec.vertices.T).T
            # o3d
            if np.abs(opt_angle) > 1e-6:
                if opt_axis == [1, 0, 0]:
                    R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([np.radians(opt_angle), 0., 0.]))
                elif opt_axis == [0, 1, 0]:
                    R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([0., np.radians(opt_angle), 0.]))
                elif opt_axis == [0, 0, 1]:
                    R = mesh_rec_o3d.get_rotation_matrix_from_xyz(np.array([0., 0., np.radians(opt_angle)]))
                mesh_rec_o3d.rotate(R, center=(0, 0, 0))



        if "ours" in REC_DIR or "SyncDreamer" in REC_DIR:
            # invert the face
            mesh_rec.invert()
            # o3d Invert the mesh faces
            mesh_rec_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_rec_o3d.triangles)[:, [0, 2, 1]])
            # Compute vertex normals to ensure correct orientation
            mesh_rec_o3d.compute_vertex_normals()




        # normalize
        mesh_rec.vertices = normalize_mesh(mesh_rec.vertices)
        vertices_rec = np.asarray(mesh_rec_o3d.vertices)
        vertices_rec = normalize_mesh(vertices_rec)
        mesh_rec_o3d.vertices = o3d.utility.Vector3dVector(vertices_rec)

        # print("mesh_gt_o3d ", np.asarray(mesh_gt_o3d.vertices).max(0), np.asarray(mesh_gt_o3d.vertices).min(0))
        # print("mesh_rec_o3d ", np.asarray(mesh_rec_o3d.vertices).max(0), np.asarray(mesh_rec_o3d.vertices).min(0))
        assert np.abs(np.asarray(mesh_gt_o3d.vertices)).max() <= 0.505
        assert np.abs(np.asarray(mesh_rec_o3d.vertices)).max() <= 0.505
        assert np.abs(np.asarray(mesh_gt.vertices)).max() <= 0.505
        assert np.abs(np.asarray(mesh_rec.vertices)).max() <= 0.505



        # compute chamfer distance
        chamfer_dist = chamfer(mesh_gt.vertices, mesh_rec.vertices)
        vol_iou = compute_iou(mesh_gt, mesh_rec)
        CDs.append(chamfer_dist)
        IoUs.append(vol_iou)

        # # todo save screenshots
        # mesh_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # # draw bbox for gt and rec
        # bbox_gt = mesh_gt.bounding_box.bounds
        # bbox_rec = mesh_rec.bounding_box.bounds
        # bbox_gt_o3d = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_gt[0], max_bound=bbox_gt[1])
        # bbox_rec_o3d = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_rec[0], max_bound=bbox_rec[1])
        # # color red for gt, green for rec
        # bbox_gt_o3d.color = (1, 0, 0)
        # bbox_rec_o3d.color = (0, 1, 0)
        # # draw a bbox of unit cube [-1, 1]^3
        # bbox_unit_cube = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(1, 1, 1))
        # bbox_unit_cube.color = (0, 0, 1)
        #
        # # o3d.visualization.draw_geometries(
        # #     [mesh_axis, mesh_gt_o3d, mesh_rec_o3d, bbox_gt_o3d, bbox_rec_o3d, bbox_unit_cube])
        #
        # # take a screenshot with circle view and save to file
        # # save screenshot to file
        # vis_output = os.path.join("screenshots", method_name)
        # os.makedirs(vis_output, exist_ok=True)
        # mesh_rec_o3d.compute_vertex_normals()
        #
        # # vis = o3d.visualization.Visualizer()
        # # vis.create_window(width=512, height=512)
        # # vis.add_geometry(mesh_rec_o3d)
        # # # show the window and save camera pose to json file
        # # vis.get_render_option().light_on = True
        # # vis.run()
        #
        # # rgb
        # for i in range(6):
        #     capture_screenshots(mesh_rec_o3d, f"ScreenCamera_{i}.json", "RenderOption_rgb.json", os.path.join(vis_output, obj_name + f"_{i}.png"))
        # # phong shading
        # for i in range(6):
        #     capture_screenshots(mesh_rec_o3d, f"ScreenCamera_{i}.json", "RenderOption_phong.json", os.path.join(vis_output, obj_name + f"_{i}_phong.png"))


        # todo 3D metrics
        # save metrics to a single file
        with open(os.path.join(REC_DIR, "metrics3D.txt"), "a") as f:
            # write metrics in one line with format: obj_name chamfer_dist volume_iou
            f.write(obj_name + " CD:" + str(chamfer_dist) + " IoU:" + str(vol_iou) + "\n")

    # average metrics and save to the file
    print("Average CD:", np.mean(CDs))
    print("Average IoU:", np.mean(IoUs))
    with open(os.path.join(REC_DIR, "metrics3D.txt"), "a") as f:
        f.write("Average CD:" + str(np.mean(CDs)) + " IoU:" + str(np.mean(IoUs)) + "\n")


### TODO
GT_DIR = "/home/xin/data/EscherNet/Data/GSO30/"
methods = {}
# methods["One2345-XL"] = ""
# methods["One2345"] = ""
# methods["PointE"] = ""
# methods["ShapeE"] = ""
# methods["DreamGaussian"] = ""
# methods["DreamGaussian-XL"] = ""
# methods["SyncDreamer"] = ""
methods["Ours_T1"] = "/GSO3D/ours_GSO_T1/NeuS/"
methods["Ours_T2"] = "/GSO3D/ours_GSO_T2/NeuS/"
methods["Ours_T3"] = "/GSO3D/ours_GSO_T3/NeuS/"
methods["Ours_T5"] = "/GSO3D/ours_GSO_T5/NeuS/"
methods["Ours_T10"] = "/GSO3D/ours_GSO_T10/NeuS"

for method_name in methods.keys():
    print("method_name: ", method_name)
    vis_3D_rec(GT_DIR, methods[method_name], method_name)
