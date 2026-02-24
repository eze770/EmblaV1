# Our implementation is based on the NeRF publicly available code from https://github.com/krrish94/nerf-pytorch/ and
# https://github.com/bmild/nerf
import random
from networks import FBV_SM, PositionalEncoder
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


def init_models(d_input, d_filter, pretrained_model_pth=None, lr=5e-4, output_size=2, FLAG_PositionalEncoder = False, return_latent=False):

    if FLAG_PositionalEncoder:
        encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)

        model = FBV_SM(encoder = encoder,
                       d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size,
                       return_latent=return_latent)

    else:
        # Models
        model = FBV_SM(d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size,
                       return_latent=return_latent)
    model.to(device)
    # Pretrained Model
    if pretrained_model_pth != None:
        model.load_state_dict(torch.load(pretrained_model_pth + "best_model.pt", map_location=torch.device(device)))
    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train(model, optimizer, different_arch, DOF, near, far, Flag_save_image_during_training, LOG_PATH, n_iters,
          chunksize, center_crop, center_crop_iters, display_rate, record_file_train, record_file_val, Patience_threshold, training_angles_snapshot, training_imges_snapshot, buffer):

    Camera_FOV = 45.
    camera_angle_y = Camera_FOV * np.pi / 180.
    focal = 0.5 * 64 / np.tan(0.5 * camera_angle_y)

    tr = 0.8  # training ratio
    buffer_size = buffer.getIndex()
    sample_id = random.sample(range(buffer_size), buffer_size)

    training_img = training_imges_snapshot[sample_id[:int(buffer_size * tr)]]
    training_angles = training_angles_snapshot[sample_id[:int(buffer_size * tr)]]

    testing_angles = training_angles_snapshot[sample_id[int(buffer_size * tr):]]
    testing_img = training_imges_snapshot[sample_id[int(buffer_size * tr):]]
    train_amount = len(training_angles)  # no usage (eze)
    valid_amount = len(testing_angles)
    print("SM Validation amount: ", valid_amount)

    max_pic_save = 6
    loss_v_last = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
    patience = 0
    min_loss = np.inf
    height, width = training_img[0].shape[1:]
    print("SM height, width: ", height, width)
    rays_o, rays_d = get_rays(height, width, focal)
    print("SM rays: ", rays_o, rays_d)

    for i in trange(n_iters):
        model.train()

        target_img_idx = np.random.randint(training_img.shape[0] - 1)

        # Pick an image as the target.
        target_img = training_img[target_img_idx]
        target_img = target_img.mean(dim=0)  # RGB → grayscale
        angle = training_angles[target_img_idx]

        if center_crop and i < center_crop_iters:
            target_img = crop_center(target_img)
            rays_o_train, rays_d_train = get_rays(int(height*0.5), int(width*0.5), focal)
        else:
            rays_o_train,rays_d_train = rays_o, rays_d

        target_img = target_img.reshape([-1])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = model_forward(rays_o_train, rays_d_train,
                               near, far, model,
                               chunksize=chunksize,
                               arm_angle=angle,
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

        # Evaluate testing at given display rate.
        if i % display_rate == 0:
            model.eval()
            torch.no_grad()
            valid_epoch_loss = []
            valid_psnr = []
            valid_image = []

            for v_i in range(valid_amount):  # valid_amount is for how much testingdata there is (faktor 0.8) (eze)
                angle = testing_angles[v_i]
                img_label = testing_img[v_i]
                img_label = img_label.mean(dim=0)

                # Run one iteration of TinyNeRF and get the rendered RGB image.
                outputs = model_forward(rays_o, rays_d,
                                       near, far, model,
                                       chunksize=chunksize,
                                       arm_angle=angle,
                                       DOF=DOF,
                                       output_flag= different_arch)

                rgb_predicted = outputs['rgb_map']

                img_label_tensor = img_label.reshape(-1).to(device)

                v_loss = torch.nn.functional.mse_loss(rgb_predicted, img_label_tensor)

                valid_epoch_loss.append(v_loss.item())

                np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
                if v_i < max_pic_save:
                    valid_image.append(np_image)
            loss_valid = np.mean(valid_epoch_loss)

            print("SM-Loss:", loss_valid, 'patience', patience)
            scheduler.step(loss_valid)

            # save test image
            np_image_combine = np.hstack(valid_image)
            np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
            np_image_combine = np.clip(np_image_combine,0,1)
            matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
            if Flag_save_image_during_training:
                matplotlib.image.imsave(LOG_PATH + '/image/' + '%d.png' % i, np_image_combine)

            record_file_train.write(str(loss_train) + "\n")
            record_file_val.write(str(loss_valid) + "\n")
            torch.save(model.state_dict(), LOG_PATH + '/best_model/model_epoch%d.pt'%i)

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

        if patience > Patience_threshold:
            break

        # torch.cuda.empty_cache()    # to save memory
    return True


def main(config, buffer):
    print("\n\nSelfmodel-training initialized...\n")
    sim_real = config.dreamer.selfModel.sim_real
    arm_ee = config.dreamer.selfModel.arm_ee
    seed_num = config.seed
    robotid = config.robotID
    FLAG_PositionalEncoder = config.dreamer.selfModel.positionalEncoder

    # 0:OM, 1:OneOut, 2: OneOut with distance
    different_arch = 0
    print('different_arch', different_arch)

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

    pxs = 100  # collected data pixels  # no usage (eze)

    # np.load('data/%s_data/%s_data_robo%d(%s).npz'%(sim_real,sim_real,robotid,arm_ee))  # needs Image, Angle, focal (eze)
    # data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d.npz'%(sim_real,sim_real,robotid,arm_ee,cam_dist*1000))
    # data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d_test.npz'%(sim_real,sim_real,robotid,arm_ee,800)) # 800 test is 1000 ... local data, Jiong

    print("DOF, robot_id, PE", DOF, robotid, FLAG_PositionalEncoder)
    LOG_PATH = "train_log/%s_id%d_(%d)_%s(%s)_%s" % (sim_real, robotid, seed_num, add_name, arm_ee, config.runName)
    config = config.dreamer.selfModel
    if different_arch != 0:
        LOG_PATH += 'diff_out_%d' % different_arch
    print("Data Loaded!")
    os.makedirs(LOG_PATH + "/image/", exist_ok=True)
    os.makedirs(LOG_PATH + "/best_model/", exist_ok=True)

    training_imges_snapshot = buffer.getData().nextObservations.clone()
    training_angles_snapshot = buffer.getData().angles.clone()
    print("SM img data size: ", len(training_imges_snapshot))
    print("SM angles data size: ", len(training_angles_snapshot))

    # Encoders
    """arm dof = 2+3; arm dof=3+3"""

    # Stratified sampling
    n_samples = config.nSamples  # Number of spatial samples per ray
    perturb = True  # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Hierarchical sampling
    n_samples_hierarchical = 64  # Number of samples per ray  # again no use (eze)
    perturb_hierarchical = False  # If set, applies noise to sample positions  # again no use (eze)

    # Training
    n_iters = config.nIters  # default = 400000 (eze)
    one_image_per_step = True  # One image per gradient step (disables batching)  # No use again (eze)
    chunksize = config.chunkSize  # Modify as needed to fit in GPU memory
    center_crop = True  # Crop the center of image (one_image_per_)   # debug
    center_crop_iters = 200  # Stop cropping center after this many epochs
    display_rate = 1000  # int(select_data_amount*tr)  # Display test output every X epochs  # only use in train (eze)

    # Early Stopping
    warmup_iters = 400  # Number of iterations during warmup phase  # no use (eze)
    warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters  # no use (eze)
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
    Patience_threshold = 100  # only use in train() (eze)

    # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
    # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'

    for _ in range(n_restarts):

        model, optimizer = init_models(d_input=(DOF - 2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                       d_filter=128,
                                       output_size=2,
                                       lr=5e-4,  # 5e-4
                                       # pretrained_model_pth=pretrained_model_pth,
                                       FLAG_PositionalEncoder=FLAG_PositionalEncoder
                                       )

        success = train(model, optimizer, different_arch, DOF, near, far, Flag_save_image_during_training, LOG_PATH,
                        n_iters, chunksize, center_crop, center_crop_iters, display_rate, record_file_train,
                        record_file_val, Patience_threshold, training_angles_snapshot, training_imges_snapshot, buffer)
        if success:
            print('Training successful!')
            break

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()


if __name__ == "__main__":
    main()

