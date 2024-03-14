import numpy as np
import os 
import cv2
import json
from PIL import Image
import math
import pytz
from datetime import datetime
import pvlib
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from justpfm import justpfm as jpfm
from ray_utils import *
from base import BaseDataset

class WaymoDataset(BaseDataset):
    def __init__(self, root_dir, split, nvs=False, downsample=1.0, load_2d=True, generate_render_path=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        self.nvs = nvs # exclude testing frames in training
        self.generate_render_path = generate_render_path


        dir_rgb = os.path.join(root_dir, 'image_0')
        dir_sem = os.path.join(root_dir, 'intrinsic_0', 'semantic')
        dir_normal = os.path.join(root_dir, 'intrinsic_0', 'normal')
        dir_shadow = os.path.join(root_dir, 'intrinsic_0', 'shadow')
        dir_deshadow = os.path.join(root_dir, 'intrinsic_0', 'deshadow_sl')
        dir_depth = os.path.join(root_dir, 'intrinsic_0', 'depth_midas')

        transform_path = os.path.join(root_dir, 'transforms.json')
        with open(transform_path, 'r') as file:
            transform = json.load(file)

        K = np.array([
            [transform['fl_x'], 0, transform['cx']],
            [0, transform['fl_y'], transform['cy'] ],
            [0, 0, 1]
        ])
        K[:2] *= downsample
        self.K = K
        w, h = int(transform['w']*downsample), int(transform['h']*downsample)
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

        # Extrinsics
        frames = transform['frames']
        ids = np.array([frames[i]['colmap_im_id'] for i in range(len(frames))])
        poses = np.array([frames[i]['transform_matrix'] for i in range(len(frames))])
        arg_sort = np.argsort(ids)
        poses = poses[arg_sort][:, :3] # (n, 3, 4) OpenGL, c2w
        poses[:, :, 1:3] *= -1  # (n, 3, 4) OpenCV, c2w

        frame_start = kwargs.get('frame_start', 0)
        frame_end   = kwargs.get('frame_end', 100)
        frame_id = np.arange(frame_start, frame_end)
        self.setup_poses(poses, frame_id)
        self.estimate_sunlight()
        
        # self.sunlight_from_2d(cam2world_0, cam2world_1, illum_0, illum_1, frame_id)
        print('#frames = {}'.format(len(frame_id)))
        print('frame_id:', frame_id)
        
        if load_2d:
            print('Load RGB ...')
            rgb = self.read_rgb(dir_rgb, frame_id)
            self.rays = torch.FloatTensor(rgb)
            if self.split == 'train':
                print('Load Semantic ...')
                sem = self.read_semantics(dir_sem, frame_id)
                self.labels = torch.LongTensor(sem)
                print('Load Normal ...')
                normal = self.read_normal(dir_normal, frame_id)
                self.normals = torch.FloatTensor(normal)
                print('Load Shadow ...')
                shadow = self.read_shadow(dir_shadow, frame_id)
                self.shadows = torch.FloatTensor(shadow)
                # print('Load Deshadow ...')
                # deshadow = self.read_rgb(dir_deshadow, frame_id)
                # self.rgb_deshadow = torch.FloatTensor(deshadow)
                print('Load Depth ...')
                depth = self.read_depth(dir_depth, frame_id)
                self.depths = torch.FloatTensor(depth)
    
    def setup_poses(self, cam2world, frame_id):
        cam2world = cam2world[frame_id]

        pos = cam2world[:, :, -1]
        forward = pos[-1] - pos[0]
        forward = forward / np.linalg.norm(forward)
        xyz_min = np.min(pos, axis=0)
        xyz_max = np.max(pos, axis=0)
        center = (xyz_min + xyz_max) / 2
        scale  = np.max(xyz_max - xyz_min) / 2
        self.scale = scale

        pos = (pos - center.reshape(1, -1)) / scale
        pos = pos - forward.reshape(1, -1) * 0.5
        cam2world[:, :, -1] = pos
        self.poses = torch.FloatTensor(cam2world)
        
        if self.generate_render_path:
            # render_c2w = generate_interpolated_path(cam2world, 120)[:400]
            render_c2w = cam2world
            self.render_c2w = torch.FloatTensor(render_c2w)
            self.render_traj_rays = self.get_path_rays(render_c2w)

    def get_path_rays(self, render_c2w):
        rays = {}
        print(f'Loading {len(render_c2w)} camera path ...')
        for idx in range(len(render_c2w)):
            c2w = np.array(render_c2w[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

    def read_rgb(self, dir_rgb, frame_id):
        rgb_list = []
        for i in frame_id:
            path = os.path.join(dir_rgb, '{:0>6d}.png'.format(i))
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_wh)
            img = (img / 255.0).astype(np.float32)
            rays = img.reshape(-1, 3)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list

    def read_depth(self, dir_depth, frame_id):
        depth_list = []
        for i in frame_id:
            # path = os.path.join(dir_depth, '{:0>6d}_depth.png'.format(i))
            # d_inv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # d_inv = np.clip(d_inv / 255, 0.05, 1)
            path = os.path.join(dir_depth, '{:0>6d}.pfm'.format(i))
            d_inv = jpfm.read_pfm(file_name=path)
            # process 
            d_inv = np.clip(d_inv/d_inv.max(), 0.05, 1)
            depth = 1 / d_inv 
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            depth = cv2.resize(depth, self.img_wh)
            depth = depth.astype(np.float32).flatten()
            depth_list.append(depth)
        depth_list = np.stack(depth_list)
        return depth_list
    
    def read_semantics(self, dir_sem, frame_id):
        label_list = []
        for i in frame_id:
            path = os.path.join(dir_sem, '{:0>6d}.pgm'.format(i))
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, self.img_wh)
            label = label.flatten()
            label_list.append(label)
        label_list = np.stack(label_list)
        return label_list
    
    def read_normal(self, dir_normal, frame_id):
        poses = self.poses.numpy()
        normal_list = []
        for c2w, i in zip(poses, frame_id):
            path = os.path.join(dir_normal, '{:0>6d}_normal.npy'.format(i))
            img = np.load(path).transpose(1, 2, 0)
            img = cv2.resize(img, self.img_wh)
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

    def read_shadow(self, dir_shadow, frame_id):
        shadow_list = []
        for i in frame_id:
            path = os.path.join(dir_shadow, '{:0>6d}.png'.format(i))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_wh)
            img = (img / 255).astype(np.float32).flatten()
            shadow_list.append(img)
        shadow_list = np.stack(shadow_list)
        return shadow_list
    
    def estimate_sunlight(self):
        self.sun_dir = torch.FloatTensor([-0.07508344, -0.05114832, -0.00162191])
        self.sun_dir /= torch.norm(self.sun_dir)
        downs = self.poses.numpy()[:, :, 1]
        self.up_dir  = torch.FloatTensor(-np.mean(downs, axis=0))
        self.up_dir /= torch.norm(self.up_dir)
        
def test():
    kwargs = {
        'frame_start': 0,
        'frame_end': 100,
        # 'downsample': 0.5,
        'test_id': [30, 40, 50, 60]
    }

    dataset = WaymoDataset(
        '../../../datasets/waymo_kitti/',
        'train', nvs=False,
        ** kwargs
    )

    dataset.ray_sampling_strategy = 'all_images'
    dataset.batch_size = 256
    print('poses:', dataset.poses.size())
    print('RGBs: ', dataset.rays.size())

    sample = dataset[0]
    print('Keys:')
    print(sample.keys())
    print('depth:' , sample['depth'].size())
    print(sample['depth'].min(), sample['depth'].max())
    # print('Sun direction:', dataset.sun_dir)
    # print('Up  direction:', dataset.up_dir)
    # print('Scene scale:', dataset.scale)

    sun_dir = dataset.sun_dir
    pose = dataset.poses[49]
    c2w_R = pose[:3, :3]
    sun_dir_cam = c2w_R.T @ sun_dir
    print('sun direction in cam:', sun_dir_cam)


    # visualize
    # import vedo
    # w, h = dataset.img_wh
    # K = dataset.K
    # poses = dataset.poses.numpy()
    # poses[:, :, -1] *= 10
    # pos = poses[:, :, -1]
    # arrow_len, s = 1, 1
    # x_end   = pos + arrow_len * poses[:, :, 0]
    # y_end   = pos + arrow_len * poses[:, :, 1]
    # z_end   = pos + arrow_len * poses[:, :, 2]
    
    # x = vedo.Arrows(pos, x_end, s=s, c='red')
    # y = vedo.Arrows(pos, y_end, s=s, c='green')
    # z = vedo.Arrows(pos, z_end, s=s, c='blue')
        
    # vedo.show(x,y,z, axes=1)

def test_undistort():

    def undistort_image(img, K, dist_coeffs):
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
        return undistorted_img, new_camera_matrix

    # Example distortion parameters
    k1, k2, p1, p2 = 0.07553952403009787, -0.43778759733518613, -6.473361187132511e-05, -0.008548148231344877

    # Example intrinsic matrix K
    fx, fy, cx, cy = 2200.899816465766, 2241.5858391355296, 932.5793290624779, 628.6263966719355
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Load an example distorted image
    image_path = '/hdd/datasets/waymo/nerfstudio/images/frame_00001.png'
    distorted_img = cv2.imread(image_path)

    # Undistort the image
    dist_coeffs = np.array([k1, k2, p1, p2])
    undistorted_img, new_camera_matrix = undistort_image(distorted_img, K, dist_coeffs)

    # Display the original and undistorted images
    # cv2.imshow('Original Image', distorted_img)
    # cv2.imshow('Undistorted Image', undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print the new intrinsic matrix K
    print("Original Intrinsic Matrix K:")
    print(K)
    print("\nNew Intrinsic Matrix K:")
    print(new_camera_matrix)

    from matplotlib import pyplot as plt
    plt.imshow(distorted_img)
    plt.show()
    plt.close()
    plt.imshow(undistorted_img)
    plt.show()
    plt.close()

def triangulate():
    from utility.triangulate import get_sample_index
    # check pixel coordinate with: https://pixspy.com/
    image_h = 1280
    image_w = 1920
    
    frame_ids = [49, 53]
    
    # pixel_coords = [
    #     [853, 171], [894, 31]
    # ]# (h, w)
    pixel_coords = [
        [787, 295], [820, 157]
    ]# (h, w)

    kwargs = {
        'frame_start': 0,
        'frame_end': 100,
        'load_2d': False
    }
    dataset = WaymoDataset(
        '/hdd/datasets/waymo/kitti/',
        'test',
        ** kwargs
    )

    directions = dataset.directions

    pose = dataset.poses[frame_ids[0]]
    rays_o, rays_d = get_rays(directions, pose)
    pixel_h, pixel_w = pixel_coords[0]
    sample_idx = get_sample_index(pixel_h, pixel_w, image_h, image_w)
    o0 = rays_o[sample_idx].numpy()
    ray0 = rays_d[sample_idx].numpy()
    pose = dataset.poses[frame_ids[1]]
    rays_o, rays_d = get_rays(directions, pose)
    pixel_h, pixel_w = pixel_coords[1]
    sample_idx = get_sample_index(pixel_h, pixel_w, image_h, image_w)
    o1 = rays_o[sample_idx].numpy()
    ray1 = rays_d[sample_idx].numpy()

    vertical = np.cross(ray0, ray1)
    normal0 = np.cross(ray0, vertical)
    normal1 = np.cross(ray1, vertical)
    s0 = np.sum(normal1 * (o1 - o0)) / np.sum(normal1 * ray0)
    s1 = np.sum(normal0 * (o0 - o1)) / np.sum(normal0 * ray1)
    p0_world = o0 + s0 * ray0
    p1_world = o1 + s1 * ray1
    point_3d = (p0_world + p1_world)/2
    print('Intersected at:', point_3d)
    print('Sun direction:', dataset.sun_dir)
    print('Up  direction:', dataset.up_dir)

def parse_camera_info(file):
    intrinsics = []
    extrinsics = []
    transform = None
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            numbers = line.split(':')[1].strip().split(' ')
            array = np.array([float(x) for x in numbers])
            if 'transform' in line:
                transform = array.reshape(4, 4)
            elif 'intrinsic' in line:
                intrinsics.append(array.reshape(3, 4)[:3, :3])
            elif 'extrinsic' in line: # extrinsics
                extrinsics.append(array.reshape(4, 4))
    intrinsics = np.stack(intrinsics).astype(np.float32)
    extrinsics = np.stack(extrinsics).astype(np.float32)
    return transform, intrinsics, extrinsics

def test_camera():
    dir_path = '/hdd/datasets/waymo/kitti_test/calib/'
    file_path = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)])

    poses = []
    int_list = []
    ext_list = []
    for file in file_path:
        pose, ints, exts = parse_camera_info(file)
        poses.append(pose)
        int_list.append(ints)
        ext_list.append(exts)

    poses = np.stack(poses)[:, :3, :]
    transform = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    poses = np.matmul(poses, transform)

    pos = poses[:, :, -1]
    xyz_min = np.min(pos, axis=0)
    xyz_max = np.max(pos, axis=0)
    center = (xyz_min + xyz_max) / 2
    scale  = np.max(xyz_max - xyz_min) / 2
    pos = (pos - center.reshape(1, -1)) / scale

    import vedo
    pos *= 10
    arrow_len, s = 1, 1
    x_end   = pos + arrow_len * poses[:, :, 0]
    y_end   = pos + arrow_len * poses[:, :, 1]
    z_end   = pos + arrow_len * poses[:, :, 2]
    
    x = vedo.Arrows(pos, x_end, s=s, c='red')
    y = vedo.Arrows(pos, y_end, s=s, c='green')
    z = vedo.Arrows(pos, z_end, s=s, c='blue')
        
    vedo.show(x,y,z, axes=1)

if __name__ == '__main__':
    test()
    # test_camera()
    # triangulate()