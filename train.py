############################################################
# SAD-GS style training + geometry losses (1~6) FULL INTEGRATED (CLS REMOVED)
#
# 1) Color loss: L1 + DSSIM                           (CS)
# 2) Depth supervised (if gt depth exists)            (DS)  [with scale+shift align]
# 3) Scale regularization (mean scaling)              (SC)
# 4) Normal consistency substitute:
#    depth->normal edge-aware smooth                  (NC)
# 5) Multi-view color consistency (warp ref->src)     (MV_C)
# 6) Multi-view depth consistency (warp ref->src)     (MV_D) [occlusion-aware via z_pred_in_ref]
#
# + Keep original SAD-GS densify/prune/reset_opacity pipeline
# + Write cfg_args (for render.py) + command.txt
#
# NOTE:
# - CLS loss removed completely.
# - MV_D is true multi-view depth consistency (not pixel reprojection geo-loss).
############################################################

import os
import sys
import time
import uuid
import numpy as np
import torch
import torch.nn.functional as F

from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr

from arguments import ModelParams, PipelineParams, OptimizationParams

# optional dist
try:
    from sklearn.neighbors import KDTree
    import open3d as o3d
    HAVE_DIST = True
except Exception:
    HAVE_DIST = False

# optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# optional wandb (avoid interactive login)
try:
    import wandb
    HAVE_WANDB = True
except Exception:
    HAVE_WANDB = False

# optional nvidia_smi
try:
    import nvidia_smi
    HAVE_NVSMI = True
except Exception:
    HAVE_NVSMI = False

CHUNK_SIZE = 50000


# =========================================================
# Utils
# =========================================================
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

def save_command(path):
    cmd = ' '.join(sys.argv)
    with open(os.path.join(path, 'command.txt'), 'a') as f:
        f.write(cmd + '\n')

def _num_or(default, x):
    if x is None:
        return float(default)
    try:
        if torch.is_tensor(x):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)

def align_scale_shift(pred_hw, gt_hw, mask_hw, eps=1e-6):
    """
    Find a,b to minimize || a*pred + b - gt || on mask.
    Returns aligned_pred, (a,b).
    """
    m = mask_hw
    if m is None:
        m = torch.ones_like(gt_hw, dtype=torch.bool)

    if m.sum().item() < 10:
        return pred_hw, (torch.tensor(1.0, device=pred_hw.device), torch.tensor(0.0, device=pred_hw.device))

    p = pred_hw[m].float()
    g = gt_hw[m].float()

    mp = p.mean()
    mg = g.mean()

    vp = (p - mp)
    cov = (vp * (g - mg)).mean()
    var = (vp * vp).mean().clamp_min(eps)

    a = cov / var
    b = mg - a * mp
    return a * pred_hw + b, (a, b)

def make_K_from_fov(cam, device):
    W = int(getattr(cam, "image_width"))
    H = int(getattr(cam, "image_height"))

    # principal point
    cx = _num_or(W * 0.5, getattr(cam, "Cx", None))
    cy = _num_or(H * 0.5, getattr(cam, "Cy", None))

    # focal: prefer Fx/Fy if present
    fx_attr = getattr(cam, "Fx", None)
    fy_attr = getattr(cam, "Fy", None)

    if fx_attr is not None and fy_attr is not None:
        fx = _num_or(W / 2.0, fx_attr)
        fy = _num_or(H / 2.0, fy_attr)
    else:
        fovx = getattr(cam, "FoVx", None)
        fovy = getattr(cam, "FoVy", None)
        if fovx is None or fovy is None:
            fx = float(max(W, H))
            fy = float(max(W, H))
        else:
            fovx = _num_or(np.deg2rad(60.0), fovx)
            fovy = _num_or(np.deg2rad(60.0), fovy)
            fx = W / (2.0 * np.tan(fovx / 2.0))
            fy = H / (2.0 * np.tan(fovy / 2.0))

    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32
    )
    return K

def get_w2c_from_RT(cam, device):
    """
    SAD-GS uses Pc = R @ Pw + T
    """
    R = cam.R
    T = cam.T
    if not torch.is_tensor(R):
        R = torch.tensor(R, device=device, dtype=torch.float32)
    else:
        R = R.to(device).float()
    if not torch.is_tensor(T):
        T = torch.tensor(T, device=device, dtype=torch.float32)
    else:
        T = T.to(device).float()
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.view(3)
    return w2c

def backproject(depth_hw, K, w2c):
    """
    depth_hw: [H,W] camera depth (Z)
    return Pw [H,W,3], valid [H,W]
    """
    device = depth_hw.device
    H, W = depth_hw.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    x = x.float()
    y = y.float()
    z = depth_hw
    valid = (z > 0) & torch.isfinite(z)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z

    Pc = torch.stack([X, Y, z, torch.ones_like(z)], dim=-1)  # [H,W,4]
    c2w = torch.linalg.inv(w2c)
    Pw = (Pc.reshape(-1, 4) @ c2w.T).reshape(H, W, 4)[..., :3]
    return Pw, valid

def project(Pw, K, w2c, H, W):
    """
    Pw: [H,W,3] -> grid [1,H,W,2] for sampling ref->src
    Returns: grid, z_pred_in_cam (Zc), valid_z, in_grid
    """
    device = Pw.device
    ones = torch.ones((H, W, 1), device=device, dtype=Pw.dtype)
    Pw4 = torch.cat([Pw, ones], dim=-1)  # [H,W,4]
    Pc4 = (Pw4.reshape(-1, 4) @ w2c.T).reshape(H, W, 4)

    Xc, Yc, Zc = Pc4[..., 0], Pc4[..., 1], Pc4[..., 2]
    valid = (Zc > 1e-6) & torch.isfinite(Zc)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (Xc / (Zc + 1e-8)) + cx
    v = fy * (Yc / (Zc + 1e-8)) + cy

    u_norm = 2.0 * (u / (W - 1)) - 1.0
    v_norm = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    in_grid = (u_norm.abs() <= 1.0) & (v_norm.abs() <= 1.0)
    return grid, Zc, valid, in_grid

def depth_to_normal(depth_hw, K):
    device = depth_hw.device
    H, W = depth_hw.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    x = x.float()
    y = y.float()
    z = depth_hw

    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z
    P = torch.stack([X, Y, z], dim=0)  # [3,H,W]

    dpx = P[:, :, 2:] - P[:, :, :-2]
    dpy = P[:, 2:, :] - P[:, :-2, :]

    dpx = dpx[:, 1:-1, :]
    dpy = dpy[:, :, 1:-1]

    n = torch.cross(dpx.permute(1, 2, 0), dpy.permute(1, 2, 0), dim=-1)  # [H-2,W-2,3]
    n = F.normalize(n, dim=-1, eps=1e-6).permute(2, 0, 1)               # [3,H-2,W-2]
    n = F.pad(n, (1, 1, 1, 1), mode='replicate')                         # [3,H,W]
    return n

def edge_weight_from_image(img_chw, k=10.0):
    gx = torch.mean(torch.abs(img_chw[:, :, 1:] - img_chw[:, :, :-1]), dim=0, keepdim=True)
    gy = torch.mean(torch.abs(img_chw[:, 1:, :] - img_chw[:, :-1, :]), dim=0, keepdim=True)
    gx = F.pad(gx, (1, 0, 0, 0), mode='replicate')
    gy = F.pad(gy, (0, 0, 1, 0), mode='replicate')
    g = gx + gy
    w = torch.exp(-k * g).clamp(0, 1)
    return w  # [1,H,W]


# =========================================================
# Logger
# =========================================================
def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for viewpoint in config['cameras']:
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def to_float(x):
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        return x.detach().float().mean().item()
    return float(x)

# =========================================================
# Training
# =========================================================
def training(dataset, opt, pipe,
             testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, args):

    import time
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    from random import randint
    from tqdm import tqdm

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, gaussians,
        single_frame_id=args.single_frame_id,
        voxel_size=args.voxel_size,
        init_w_gaussian=args.init_w_gaussian,
        load_ply=args.load_ply
    )

    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, device="cuda", dtype=torch.float32)

    # ---------------- timing (must exist) ----------------
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)
    total_computing_time = 0.0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack_all = scene.getTrainCameras().copy()

    # ---------------- dist (optional) ----------------
    if args.dist:
        raw_pc_map = []
        for v in viewpoint_stack_all:
            raw_pc_map.append(v.raw_pc)
        raw_pc_map = np.concatenate(raw_pc_map, axis=0)
        kdtree = KDTree(raw_pc_map, leaf_size=1)
    else:
        raw_pc_map = None
        kdtree = None

    # ======================================================
    # main loop
    # ======================================================
    for iteration in range(first_iter + 1, opt.iterations + 1):

        iter_start.record()
        tic = time.time()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # ================= GT =================
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = getattr(viewpoint_cam, "depth", None)

        # ================= Render =================
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # ================= Loss holders =================
        loss = torch.zeros((), device=image.device)
        loss_color = torch.zeros_like(loss)
        loss_depth = torch.zeros_like(loss)
        loss_alpha = torch.zeros_like(loss)
        loss_sc = torch.zeros_like(loss)
        loss_nc = torch.zeros_like(loss)
        loss_mvc = torch.zeros_like(loss)
        loss_mvd = torch.zeros_like(loss)
        loss_dist = torch.zeros_like(loss)

        # ======================================================
        # 1) Color loss (CS)
        # ======================================================
        Ll1 = l1_loss(image, gt_image)
        loss_color = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if args.CS:
            loss += args.CS * loss_color


        # ======================================================
        # 2) Depth (ABSOLUTE MINIMAL)
        #   - no align, no mask, no max_depth, no sanitize
        #   - only clamp negative gt to 0 (跟你 minimal 一样)
        # ======================================================
        loss_depth = torch.tensor(0.0, device=image.device)

        if (args.DS is not None) and (args.DS > 0) and (gt_depth is not None):
            gt_depth_ = gt_depth
            if not torch.is_tensor(gt_depth_):
                gt_depth_ = torch.tensor(gt_depth_, device=image.device)
            gt_depth_ = gt_depth_.to(image.device).float()

            # make sure gt is [H,W]
            if gt_depth_.dim() == 3:
                gt_depth_ = gt_depth_[0]

            # renderer depth is usually [1,H,W]; keep it as-is (跟 minimal 一样 clone)
            pred_depth = depth.clone()

            # minimal clamp
            gt_depth_[gt_depth_ < 0] = 0

            # EXACT minimal computation
            depth_term = l1_loss(pred_depth, gt_depth_)

            # warmup (跟 minimal 一样)
            if getattr(args, "depth_warmup_iters", 0) > 0 and iteration < args.depth_warmup_iters:
                w = args.DS * float(iteration) / float(args.depth_warmup_iters)
            else:
                w = args.DS

            loss += depth_term * w
            loss_depth = depth_term * w


        # ======================================================
        # 3) Alpha loss（如果你开）
        # ======================================================
        if args.alpha_loss and gt_depth is not None:
            gt_d = gt_depth.to(image.device)
            if gt_d.dim() == 3:
                gt_d = gt_d[0]
            a = alpha.clone()
            gt_a = torch.ones_like(a)
            gt_a[0][gt_d == 0] = 0
            a[0][gt_d != 0] = 1
            loss_alpha = l1_loss(gt_a, a)
            loss += args.alpha_loss * loss_alpha

        if args.SC and hasattr(gaussians, "get_scaling"):
            # ---- scale floor to protect density ----
            # 建议：和 scene 尺度相关，而不是固定常数
            min_scale = args.sc_min if hasattr(args, "sc_min") else 0.01 * scene.cameras_extent

            scale = gaussians.get_scaling
            scale_clamped = torch.clamp(scale, min=min_scale)

            loss_sc = scale_clamped.mean()
            loss += args.SC * loss_sc

        # ======================================================
        # 4) Normal consistency substitute (depth-normal edge-aware smooth)
        # ======================================================
        if args.NC is not None and args.NC > 0:
            d_hw = depth[0] if depth.dim() == 3 else depth
            d_hw = d_hw.clone()
            d_hw[~torch.isfinite(d_hw)] = 0
            d_hw[d_hw < 0] = 0
            if args.nc_max_depth is not None:
                d_hw[d_hw > args.nc_max_depth] = 0

            K = make_K_from_fov(viewpoint_cam, device=d_hw.device)
            n = depth_to_normal(d_hw, K)
            w = edge_weight_from_image(gt_image, k=args.NC_edge).detach()

            nx = (n[:, :, 1:] - n[:, :, :-1]).abs().mean(0, keepdim=True)
            ny = (n[:, 1:, :] - n[:, :-1, :]).abs().mean(0, keepdim=True)
            nx = F.pad(nx, (1, 0, 0, 0), mode='replicate')
            ny = F.pad(ny, (0, 0, 1, 0), mode='replicate')
            loss_nc = (w * (nx + ny)).mean()

            loss = loss + args.NC * loss_nc

        # ======================================================
        # 5/6) Multi-view consistency
        #   MVC: photometric (warp ref->src)
        #   MVD: depth consistency (warp ref depth->src) with occlusion-aware gating using z_pred_in_ref
        # ======================================================
        do_mv = (iteration >= args.mv_start_iter) and ((iteration % args.mv_interval) == 0)
        if getattr(args, "mv_end_iter", 0) and args.mv_end_iter > 0:
            do_mv = do_mv and (iteration <= args.mv_end_iter)

        if do_mv and (len(viewpoint_stack_all) > 1) and ((args.MV_C and args.MV_C > 0) or (args.MV_D and args.MV_D > 0)):
            d_src = depth[0] if depth.dim() == 3 else depth
            d_src = d_src.clone()
            d_src[~torch.isfinite(d_src)] = 0
            d_src[d_src < 0] = 0

            if args.mv_max_depth is not None:
                d_src[d_src > args.mv_max_depth] = 0

            H, W = d_src.shape
            K_src = make_K_from_fov(viewpoint_cam, device=d_src.device)
            w2c_src = get_w2c_from_RT(viewpoint_cam, device=d_src.device)

            Pw, v0 = backproject(d_src, K_src, w2c_src)

            mv_c_acc = 0.0
            mv_d_acc = 0.0
            used_c = 0
            used_d = 0

            for _ in range(args.mv_pairs):
                ref_cam = viewpoint_stack_all[randint(0, len(viewpoint_stack_all) - 1)]
                if ref_cam is viewpoint_cam:
                    continue

                ref_pkg = render(ref_cam, gaussians, pipe, background)
                ref_img = ref_pkg["render"].clamp(0, 1)
                ref_dep = ref_pkg["depth"]
                ref_dep = ref_dep[0] if ref_dep.dim() == 3 else ref_dep
                ref_dep = ref_dep.clone()
                ref_dep[~torch.isfinite(ref_dep)] = 0
                ref_dep[ref_dep < 0] = 0
                if args.mv_max_depth is not None:
                    ref_dep[ref_dep > args.mv_max_depth] = 0

                K_ref = make_K_from_fov(ref_cam, device=d_src.device)
                w2c_ref = get_w2c_from_RT(ref_cam, device=d_src.device)

                grid, z_ref_pred, v1, in_grid = project(Pw, K_ref, w2c_ref, H, W)

                # warp ref -> src grid
                ref_img_warp = F.grid_sample(ref_img.unsqueeze(0), grid, align_corners=True).squeeze(0)
                ref_dep_warp = F.grid_sample(ref_dep[None, None], grid, align_corners=True).squeeze(0).squeeze(0)

                # sanitize warped
                ref_dep_warp = ref_dep_warp.clone()
                ref_dep_warp[~torch.isfinite(ref_dep_warp)] = 0
                ref_dep_warp[ref_dep_warp < 0] = 0
                if args.mv_max_depth is not None:
                    ref_dep_warp[ref_dep_warp > args.mv_max_depth] = 0

                # base valid
                m = v0 & v1 & in_grid & (d_src > 0)

                # MVC mask
                if m.sum().item() >= args.mv_min_valid and args.MV_C and args.MV_C > 0:
                    mv_c_acc = mv_c_acc + charbonnier((image - ref_img_warp)[:, m]).mean()
                    used_c += 1

                # MVD mask (need ref depth + occlusion-aware gating)
                if args.MV_D and args.MV_D > 0:
                    md = m & (ref_dep_warp > 0) & (z_ref_pred > 0) & torch.isfinite(z_ref_pred)

                    # occlusion-aware gate (this is the correct pair to gate on)
                    if args.mv_occ_th is not None and args.mv_occ_th > 0:
                        md = md & (torch.abs(ref_dep_warp - z_ref_pred) < args.mv_occ_th)

                    if md.sum().item() >= args.mv_min_valid:
                        if args.mv_align_depth:
                            # optional: align ref_dep_warp -> z_ref_pred (same coordinate system: ref camera Z)
                            ref_aligned, _ = align_scale_shift(ref_dep_warp, z_ref_pred, md)
                            mv_d_acc = mv_d_acc + charbonnier(ref_aligned[md] - z_ref_pred[md]).mean()
                        else:
                            mv_d_acc = mv_d_acc + charbonnier(ref_dep_warp[md] - z_ref_pred[md]).mean()
                        used_d += 1

            if used_c > 0 and args.MV_C and args.MV_C > 0:
                loss_mvc = mv_c_acc / used_c
                loss = loss + args.MV_C * loss_mvc

            if used_d > 0 and args.MV_D and args.MV_D > 0:
                loss_mvd = mv_d_acc / used_d
                loss = loss + args.MV_D * loss_mvd

        # ======================================================
        # dist loss (optional)
        # ======================================================
        if args.dist:
            lambda_distloss = args.dist_w
            distances, indices = kdtree.query(gaussians.get_xyz.float().detach().cpu().numpy())
            indices = indices[:, 0]
            corr_pc = torch.tensor(raw_pc_map[indices], device=image.device, dtype=torch.float32)
            thres = args.dist_th
            dist_loss = ((torch.relu(torch.norm(gaussians.get_xyz - corr_pc, dim=1) - thres)) ** 2).mean()
            loss_dist = dist_loss * lambda_distloss
            loss = loss + loss_dist

        # ======================================================
        # Opacity reset tricks (kept)
        # ======================================================
        margin_scale = 1.0

        if (args.reset_opa_far or args.reset_opa_near) and iteration > 1 and iteration % 100 == 0 and gt_depth is not None:
            camera_pose = torch.tensor(viewpoint_cam.mat).float().cuda()
            projmatrix = viewpoint_cam.get_projection_matrix().float().cuda()
            thres = 0.05 * margin_scale
            gamma = 0.001
            if (not args.reset_opa_far) and args.reset_opa_near:
                gaussians.reset_opacity_by_depth_image_fast(
                    camera_pose, projmatrix,
                    gt_depth.shape[1], gt_depth.shape[0],
                    viewpoint_cam.Cx, viewpoint_cam.Cy,
                    gt_depth.unsqueeze(0), thres, gamma, near_far=False
                )
            elif args.reset_opa_far and args.reset_opa_near:
                gaussians.reset_opacity_by_depth_image_fast(
                    camera_pose, projmatrix,
                    gt_depth.shape[1], gt_depth.shape[0],
                    viewpoint_cam.Cx, viewpoint_cam.Cy,
                    gt_depth.unsqueeze(0), thres, gamma, near_far=True
                )
            else:
                print('Error reset_opa flags')
                sys.exit()

        if args.fov_mask and iteration > 1 and iteration % 100 == 0:
            camera_pose = torch.tensor(viewpoint_cam.mat).float().cuda()
            projmatrix = viewpoint_cam.get_projection_matrix().float().cuda()
            gamma = 0.001
            gaussians.reset_opacity_outside_fov(camera_pose, projmatrix, image.shape[2], image.shape[1], gamma)

        if args.full_reset_opa and iteration % 100 == 0 and gt_depth is not None:
            thres = 0.05 * margin_scale
            gamma = 0.001
            preserve_mask = None
            for view_cam_ in viewpoint_stack_all:
                camera_pose = torch.tensor(view_cam_.mat).float().cuda()
                projmatrix = view_cam_.get_projection_matrix().float().cuda()
                gt_depth_ = view_cam_.depth

                reset_depth_mask_ = gaussians.mask_by_depth_image(
                    camera_pose, projmatrix,
                    gt_depth_.shape[1], gt_depth_.shape[0],
                    gt_depth_.unsqueeze(0),
                    thres, near=True, far=True
                )
                reset_fov_mask_ = gaussians.mask_outside_fov(
                    camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0]
                ).view(-1, 1)

                reset_mask_ = reset_depth_mask_ + reset_fov_mask_
                preserve_mask_ = ~reset_mask_
                if preserve_mask is None:
                    preserve_mask = preserve_mask_
                else:
                    preserve_mask += preserve_mask_

            gaussians.reset_opacity_by_mask(~preserve_mask, gamma)

        # ======================================================
        # Backward
        # ======================================================
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # -----------------------------
            # EMA + progress
            # -----------------------------
            ema_loss_for_log = 0.4 * float(loss.item()) + 0.6 * float(ema_loss_for_log)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            # -----------------------------
            # periodic print
            # -----------------------------
            if iteration % 100 == 0:
                print(
                    f"[{iteration:06d}] "
                    f"total={to_float(loss):.6f} | "
                    f"color(CS={args.CS:g})={to_float(loss_color):.6f} "
                    f"ds(DS={0 if args.DS is None else args.DS:g})={to_float(loss_depth):.6f} "
                    f"alpha(AL={0 if args.alpha_loss is None else args.alpha_loss:g})={to_float(loss_alpha):.6f} | "
                    f"sc(SC={args.SC:g})={to_float(loss_sc):.6f} "
                    f"nc(NC={args.NC:g})={to_float(loss_nc):.6f} | "
                    f"mvc(MV_C={args.MV_C:g})={to_float(loss_mvc):.6f} "
                    f"mvd(MV_D={args.MV_D:g})={to_float(loss_mvd):.6f} | "
                    f"dist={to_float(loss_dist):.6f} "
                    f"(mv_used={'yes' if do_mv else 'no'})"
                )

            # -----------------------------
            # wall time + cuda timing
            # -----------------------------
            toc = time.time()
            total_computing_time += (toc - tic)

            # ensure CUDA events are ready before reading elapsed_time
            torch.cuda.synchronize()
            iter_ms = iter_start.elapsed_time(iter_end)

            # -----------------------------
            # eval / tb
            # -----------------------------
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_ms, testing_iterations, scene, render, (pipe, background)
            )

            # -----------------------------
            # save gaussians
            # -----------------------------
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ======================================================
            # Densify / Prune (SAD-GS core + your vacuum/guard)
            # ======================================================
            if iteration < opt.densify_until_iter:
                # SAD-GS stats
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # densify interval
                if iteration > opt.densify_from_iter and (iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune_original(
                        opt.densify_grad_threshold,
                        float(args.prune_low_alpha),
                        scene.cameras_extent,
                        size_threshold
                    )

                # opacity reset 只在 densify 阶段做（后期不频繁 reset）
                if (iteration % opt.opacity_reset_interval == 0) or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset_opacity !!! (densify stage)")
                    gaussians.reset_opacity()

            # ======================================================
            # Optim step
            # ======================================================
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ======================================================
            # Checkpoint
            # ======================================================
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )


    progress_bar.close()
    print("\nTraining complete.")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # -------------------------
    # Loss weights (1~6 default ON)
    # -------------------------
    parser.add_argument("--CS", type=float, default=1.0)
    parser.add_argument("--DS", type=float, default=0.3)
    parser.add_argument("--alpha_loss", type=float, default=0.0)

    parser.add_argument("--SC", type=float, default=1e-4)
    parser.add_argument("--NC", type=float, default=0.01)
    parser.add_argument("--MV_C", type=float, default=0.03)
    parser.add_argument("--MV_D", type=float, default=0.03)

    # NC config
    parser.add_argument("--NC_edge", type=float, default=10.0)
    parser.add_argument("--nc_max_depth", type=float, default=10.0)

    # DS config (fixed)
    parser.add_argument("--ds_align_scale_shift", action="store_true", default=True)
    parser.add_argument("--ds_min_valid", type=int, default=2000)
    parser.add_argument("--ds_max_depth", type=float, default=10.0)

    # MV config  (RARE by default)
    parser.add_argument("--mv_start_iter", type=int, default=2000)   # 2000后才允许
    parser.add_argument("--mv_interval", type=int, default=5)       # 每20步才尝试一次
    parser.add_argument("--mv_pairs", type=int, default=1)
    parser.add_argument("--mv_max_depth", type=float, default=10.0)
    parser.add_argument("--mv_min_valid", type=int, default=6000)    # 更严格，减少“算出来”的次数
    parser.add_argument("--mv_end_iter", type=int, default=12000)  # 0/负数=不截止

    # MV depth consistency (occlusion-aware)
    parser.add_argument("--mv_occ_th", type=float, default=0.05)         # meters-ish; tune 0.02~0.1
    parser.add_argument("--mv_align_depth", action="store_true", default=False)

    # -------------------------
    # Optional dist loss
    # -------------------------
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--dist_w", type=float, default=1e1)
    parser.add_argument("--dist_th", type=float, default=0.0)

    # -------------------------
    # Opacity tricks
    # -------------------------
    parser.add_argument("--reset_opa_far", action="store_true", default=False)
    parser.add_argument("--reset_opa_near", action="store_true", default=False)
    parser.add_argument("--fov_mask", action="store_true", default=False)
    parser.add_argument("--full_reset_opa", action="store_true", default=False)
    # -------------------------
    # Densify / prune extras
    # -------------------------
    parser.add_argument("--prune_low_alpha", type=float, default=0.005)
    parser.add_argument("--hard_vacuum_until", type=int, default=8000)
    parser.add_argument("--vacuum_alpha", type=float, default=0.01)
    parser.add_argument("--prune_min_points", type=int, default=0)  # 0 = disable guard
    parser.add_argument("--depth_warmup_iters", type=int, default=2000)


    # -------------------------
    # Scene options
    # -------------------------
    parser.add_argument("--load_ply", action="store_true", default=False)
    parser.add_argument("--init_w_gaussian", action="store_true", default=False)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--single_frame_id", type=list_of_ints, default=[])

    # misc
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--TUM", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # wandb (never interactive)
    if args.wandb and HAVE_WANDB:
        # if you export WANDB_MODE=disabled, wandb will stay disabled anyway.
        mode = os.environ.get("WANDB_MODE", "online")
        name = args.model_path.split('/')[-1]
        wandb.init(project='gaussian_splatting', name=name, config={}, save_code=True, notes="", mode=mode)
    elif args.wandb and (not HAVE_WANDB):
        print("[WARN] args.wandb=True but wandb not installed; disabling wandb.")

    if HAVE_NVSMI:
        try:
            nvidia_smi.nvmlInit()
        except Exception:
            pass

    os.makedirs(args.model_path, exist_ok=True)
    save_command(args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from, args
    )

    if HAVE_NVSMI:
        try:
            nvidia_smi.nvmlShutdown()
        except Exception:
            pass

    print("\nTraining complete.")
