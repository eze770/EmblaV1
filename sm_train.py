# Our implementation is based on the NeRF publicly available code from https://github.com/krrish94/nerf-pytorch/ and
# https://github.com/bmild/nerf
import random

import torch

from func import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("train,", device)


def crop_center(
        img: torch.Tensor,
        frac: float = 0.5
) -> torch.Tensor:
    r"""
  Crop center square from image.
  """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]


def train(model, optimizer, different_arch, DOF, near, far, Flag_save_image_during_training, LOG_PATH,
          center_crop, center_crop_iters, record_file_train, record_file_val, Patience_threshold, data, config, totalGradientSteps):

    Camera_FOV = 45.
    camera_angle_y = Camera_FOV * np.pi / 180.
    focal = 0.5 * 64 / np.tan(0.5 * camera_angle_y)
    tr = 0.8  # training ratio
    batchLength = config.batchLength
    batchSize = config.batchSize
    train_amount = int(batchLength * batchSize * tr)

    training_imges_snapshot = data.nextObservations.clone()
    training_angles_snapshot = data.angles.clone()

    training_angles_snapshot = training_angles_snapshot.reshape(-1, training_angles_snapshot.shape[2])
    training_imges_snapshot = training_imges_snapshot.reshape(-1, training_imges_snapshot.shape[2], training_imges_snapshot.shape[3], training_imges_snapshot.shape[4])

    training_img = training_imges_snapshot[:train_amount]
    training_angles = training_angles_snapshot[:train_amount]

    testing_angles = training_angles_snapshot[train_amount:]
    testing_img = training_imges_snapshot[train_amount:]

    print("SM img data size: ", train_amount)
    print("SM angles data size: ", train_amount)
    print("SM Validation amount: ", int(batchLength * 0.2 * batchSize))

    max_pic_save = 6
    loss_v_last = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
    patience = 0
    min_loss = np.inf
    height, width = training_img[0].shape[1:]
    print("SM height, width: ", height, width)
    rays_o, rays_d = get_rays(height, width, focal)
    print("SM rays: ", rays_o, rays_d)

    chunksize = eval(config.selfModel.chunkSize)  # Modify as needed to fit in GPU memory
    display_rate = config.selfModel.displayRate  # int(select_data_amount*tr)  # Display test output every X epochs
    latents = torch.zeros(batchSize, batchLength-1, config.selfModel.d_filter//4, device=device)  # batchLength -1 because WM ignores first fullstate, (eze)


    model.train()

    # Pick an image as the target. # RGB → grayscale, (eze)
    if training_img.shape[1] in (1, 3):  # Channel first
        target_img = training_img.mean(dim=1)
    else:  # Channel last
        target_img = training_img.mean(dim=-1)
    """
    if center_crop:
        target_img[:center_crop_iters] = crop_center(target_img[:center_crop_iters])
        rays_o_train, rays_d_train = get_rays(int(height*0.5), int(width*0.5), focal)
    else:
        rays_o_train,rays_d_train = rays_o, rays_d
        """  # not funktional because rays are fix, (eze)

    rays_o_train, rays_d_train = rays_o, rays_d
    target_img = target_img.reshape([-1])

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    outputs, latents_train = model_forward(rays_o_train, rays_d_train,
                            near, far, model,
                            chunksize=chunksize,
                            arm_angle=training_angles,
                            DOF=DOF,
                            output_flag= different_arch)

    # Backprop!
    rgb_predicted = outputs['rgb_map']
    optimizer.zero_grad()
    target_img = target_img.to(device)
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    loss_train = loss.item()


    # Evaluate testing
    model.eval()
    torch.no_grad()
    valid_psnr = []
    valid_image = []

    if training_img.shape[1] in (1, 3):  # Channel first
        img_label = testing_img.mean(dim=1)
    else:  # Channel last
        img_label = testing_img.mean(dim=-1)

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    outputs, latents_val = model_forward(rays_o, rays_d,
                            near, far, model,
                            chunksize=chunksize,
                            arm_angle=testing_angles,
                            DOF=DOF,
                            output_flag=different_arch)

    latents = torch.cat((latents_train, latents_val), dim=0)

    rgb_predicted = outputs['rgb_map']
    img_label_tensor = img_label.reshape(-1)
    v_loss = torch.nn.functional.mse_loss(rgb_predicted, img_label_tensor)
    np_image = rgb_predicted.reshape([-1, height, width, 1]).detach().cpu().numpy()
    valid_image.append(np_image[:7])
    loss_valid = np.mean(v_loss.item())

    print("SM-Loss:", loss_valid, 'patience', patience)
    scheduler.step(loss_valid)

    # save test image
    np_image_combine = np.hstack(valid_image)
    np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
    np_image_combine = np.clip(np_image_combine,0,1)
    matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
    if Flag_save_image_during_training:
        matplotlib.image.imsave(LOG_PATH + '/image/' + '%d.png' %((totalGradientSteps + 1) * config.batchSize * config.batchLength), np_image_combine)

    record_file_train.write(str(loss_train) + "\n")
    record_file_val.write(str(loss_valid) + "\n")
    torch.save(model.state_dict(), LOG_PATH + '/best_model/model_epoch%d.pt'%((totalGradientSteps + 1) * 1000))

    if min_loss > loss_valid:
        """record the best image and model"""
        min_loss = loss_valid
        matplotlib.image.imsave(LOG_PATH + '/image/' + 'best.png', np_image_combine)
        torch.save(model.state_dict(), LOG_PATH + '/best_model/best_model.pt')
        patience = 0
    elif loss_valid == loss_v_last:
        print("restart")
        return False
    else:
        patience += 1
    loss_v_last = loss_valid
    # os.makedirs(LOG_PATH + "epoch_%d_model" % i, exist_ok=True)
    # torch.save(model.state_dict(), LOG_PATH + 'epoch_%d_model/nerf.pt' % i)
    """
    if patience > Patience_threshold:
        break"""  # not used in batch version, (eze)

    # torch.cuda.empty_cache()    # to save memory
    return True, latents


def main(config, data, totalGradientSteps):
    print("\n\nSelfmodel-training initialized...\n")
    sim_real = config.dreamer.selfModel.sim_real
    arm_ee = config.dreamer.selfModel.arm_ee
    seed_num = config.seed
    robotid = config.robotID
    FLAG_PositionalEncoder = config.dreamer.selfModel.positionalEncoder

    # 0:OM, 1:OneOut, 2: OneOut with distance, 4: Output with latents
    different_arch = 4

    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)

    DOF = config.dreamer.selfModel.dof  # the number of motors  # dof4 apr03

    cam_dist = config.dreamer.selfModel.camDist
    nf_size = config.dreamer.selfModel.nfSize
    near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
    Flag_save_image_during_training = True

    if FLAG_PositionalEncoder:
        add_name = 'PE'
    else:
        add_name = 'no_PE'

    LOG_PATH = "train_log/%s_id%d_(%d)_%s(%s)_%s" % (sim_real, robotid, seed_num, add_name, arm_ee, config.runName)

    if totalGradientSteps == 0:
        pretrained_model_pth = LOG_PATH + '/best_model/best_model.pt'
    else:
        pretrained_model_pth = LOG_PATH + '/best_model/model_epoch%d.pt'%(totalGradientSteps * 1000)

    config = config.dreamer
    #if different_arch != 0:
    #    LOG_PATH += 'diff_out_%d' % different_arch
    print("Data Loaded!")
    os.makedirs(LOG_PATH + "/image/", exist_ok=True)
    os.makedirs(LOG_PATH + "/best_model/", exist_ok=True)

    # Encoders
    """arm dof = 2+3; arm dof=3+3"""

    # Stratified sampling
    n_samples = config.selfModel.nSamples  # Number of spatial samples per ray
    perturb = True  # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Hierarchical sampling
    n_samples_hierarchical = 64  # Number of samples per ray  # again no use (eze)
    perturb_hierarchical = False  # If set, applies noise to sample positions  # again no use (eze)

    # Training
    n_iters = config.selfModel.nIters  # default = 400000 (eze)
    one_image_per_step = True  # One image per gradient step (disables batching)  # No use again (eze)
    if totalGradientSteps == 0:
        center_crop = True  # Crop the center of image (one_image_per_)   # debug
        center_crop_iters = 200  # Stop cropping center after this many epochs
    else:
        center_crop = False
        center_crop_iters = 0

    # Early Stopping
    warmup_iters = 400  # Number of iterations during warmup phase
    warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
    n_restarts = 1000  # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.  # no use (eze)
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {  # no use (eze)
        'perturb': perturb
    }

    record_file_train = open(LOG_PATH + "/log_train.txt", "w")
    record_file_val = open(LOG_PATH + "/log_val.txt", "w")
    Patience_threshold = 100

    # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
    # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'

    for _ in range(n_restarts):

        model, optimizer = init_models(d_input=(DOF - 2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                       d_filter=config.selfModel.d_filter,
                                       output_size=2,
                                       lr=5e-4,  # 5e-4
                                       pretrained_model_pth=pretrained_model_pth,
                                       FLAG_PositionalEncoder=FLAG_PositionalEncoder,
                                       return_latent=True
                                       )

        success, latents = train(model, optimizer, different_arch, DOF, near, far, Flag_save_image_during_training, LOG_PATH,
                                 center_crop, center_crop_iters, record_file_train,
                                 record_file_val, Patience_threshold, data, config, totalGradientSteps)
        if success:
            print('SelfModel-Training successful!')
            break

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()

    return latents


if __name__ == "__main__":
    main()

