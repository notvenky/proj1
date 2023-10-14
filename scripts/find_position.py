import cv2

import mujoco
import glfw
import numpy as np
import matplotlib.pyplot as plt


def make_visuals(model,
                 width=640,
                 height=480,
                 camid=3,
                 title='Wriggly Colored'):
    glfw.init()
    window = glfw.create_window(width, height, title, None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
    viewport.width, viewport.height = glfw.get_framebuffer_size(window)

    scn = mujoco.MjvScene(model, maxgeom=10000)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.fixedcamid = camid

    return window, viewport, scn, ctx, vopt, pert, cam


def get_frame(model, data, vopt, pert, cam):
    mujoco.mjv_updateScene(model, data, vopt, pert, cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, scn)

    mujoco.mjr_render(viewport, scn, ctx)
    shape = glfw.get_framebuffer_size(window)
    rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
    depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
    mujoco.mjr_readPixels(rgb_img, depth_img, viewport, ctx)

    rgb_img = np.flipud(rgb_img).copy()
    depth_img = np.flipud(depth_img).copy()
    # convert depth to meters
    extent = model.stat.extent
    z_near = model.vis.map.znear * extent
    z_far = model.vis.map.zfar * extent
    depth_img = z_near / (1 - depth_img * (1 - z_near / z_far))
    return rgb_img, depth_img


COLOR_HSV_RANGES = {
    'red': [(0, 100, 60), (10, 255, 255)],
    'green': [(40, 100, 60), (80, 255, 255)],
    'blue': [(100, 100, 60), (140, 255, 255)],
    'yellow': [(25, 100, 60), (40, 255, 255)],
    'orange': [(10, 100, 50), (25, 255, 255)],
    'purple': [(140, 100, 40), (160, 255, 255)],
}

COLOR_SITE_MAP = {
    'green': 0,
    'yellow': 1,
    'purple': 2,
    'blue': 3,
    'orange': 4,
    'red': 5
}
SITE_COLOR_MAP = {v: k for k, v in COLOR_SITE_MAP.items()}

COLOR_RGB = {
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'purple': (255, 0, 255),
    'blue': (0, 0, 255),
    'orange': (255, 128, 0),
    'red': (255, 0, 0)
}


def convert_to_hsv(r, g, b):
    rgb = np.uint8([[[r, g, b]]])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def find_colors(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    detected_colors = {}

    for k in COLOR_HSV_RANGES:
        low, high = COLOR_HSV_RANGES[k]
        mask = cv2.inRange(hsv, low, high)
        nonzero_x, nonzero_y = np.where(mask != 0)
        if len(nonzero_x) == 0:
            detected_colors[k] = None
        else:
            detected_colors[k] = (nonzero_x.mean(), nonzero_y.mean())
    return detected_colors


def plot_colors(rgb, detected_colors):
    for color in detected_colors:
        position = detected_colors[color]
        if position is None:
            continue
        rgb = cv2.circle(rgb, (int(position[1]), int(position[0])), 5,
                         COLOR_RGB[color], 1)
    return rgb


def plot_colors_depth(depth, detected_colors):
    depth_min = depth.min()
    depth_max = depth.max()
    depth_uint = np.uint8((depth - depth_min) / (depth_max - depth_min) * 255)
    for color in detected_colors:
        position = detected_colors[color]
        if position is None:
            continue
        depth_uint = cv2.circle(depth_uint,
                                (int(position[1]), int(position[0])), 5,
                                (255, 255, 255), 1)
    # convert back to float
    depth_circled = depth_uint / 255 * (depth_max - depth_min) + depth_min
    return depth_circled


def random_quat():
    x1, x2, x3 = np.random.rand(3)
    x = np.sqrt(1 - x1) * np.sin(2 * np.pi * x2)
    y = np.sqrt(1 - x1) * np.cos(2 * np.pi * x2)
    z = np.sqrt(x1) * np.sin(2 * np.pi * x3)
    w = np.sqrt(x1) * np.cos(2 * np.pi * x3)
    return np.array([w, x, y, z])


def pixel_to_camera(u, v, z, K_inv):
    pixel_coords = np.array([u, v, 1, 1 / z])
    camera_coords = (K_inv @ pixel_coords) * z
    return camera_coords[:3]


if __name__ == '__main__':
    height = 1080
    width = 1920
    cam_id = 3
    model = mujoco.MjModel.from_xml_path('../wriggly_train/envs/wriggly_mujoco/wriggly_colored.xml')
    data = mujoco.MjData(model)
    window, viewport, scn, ctx, vopt, pert, cam = make_visuals(model,
                                                               width=width,
                                                               height=height,
                                                               camid=cam_id)

    data.qpos[3:7] = random_quat()
    data.qpos[2] = 0.5
    data.qpos[7:] = np.random.randn(5)
    mujoco.mj_step(model, data)
    rgb_img, depth_img = get_frame(model, data, vopt, pert, cam)
    detected_colors = find_colors(rgb_img)

    fovy = model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(fovy * np.pi / 360)
    K = np.array(((-f, 0, width / 2, 0), (0, f, height / 2, 0), (0, 0, 1, 0),
                  (0, 0, 0, 1)))
    U, V = np.meshgrid(np.arange(height), np.arange(width))
    U = U.flatten()
    V = V.flatten()

    depth_flat = depth_img[U, V]
    # pixel_coords = np.stack((U, V, np.ones_like(U)), axis=1)
    pixel_first_three = np.stack((U, V, np.ones_like(U)), axis=1)
    invd = (1 / depth_flat).reshape(-1, 1)
    pixel_coords = np.concatenate((pixel_first_three, invd), axis=1)
    K_inv = np.linalg.inv(K)
    unscaled = pixel_coords @ K_inv.T
    camera_coords = unscaled * depth_flat
    camera_xyz = camera_coords[:, :3].reshape(height, width, 3)

    # plt.imshow(depth_img)
    # plt.show()
    detected_positions_pixel = {}
    detected_positions_camera = {}
    for color in detected_colors:
        position = detected_colors[color]
        if position is None:
            detected_positions_pixel[color] = None
            detected_positions_camera[color] = None
            continue
        u, v = position
        z = depth_img[int(u), int(v)].item()
        detected_positions_pixel[color] = np.array([u, v, z])
        detected_positions_camera[color] = pixel_to_camera(u, v, z, K_inv)

    circled = plot_colors(rgb_img, detected_colors)
    depth_circled = plot_colors_depth(depth_img, detected_colors)

    print(detected_positions_camera)
    plt.imshow(circled)
    plt.figure()
    plt.imshow(depth_circled)
    plt.show()