import torch
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import imageio
import matplotlib
import matplotlib.image
import random
import time
import cv2

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderConv, DecoderConv, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
from func import *


class Dreamer:
    def __init__(self, observationShape, actionSize, actionLow, actionHigh, dt, device, config, configFile):
        self.observationShape   = observationShape
        self.actionSize         = actionSize
        self.dt                 = dt
        self.config             = config
        self.configFile         = configFile
        self.device             = device

        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.smLatentSize   = config.selfModel.d_filter // 4
        self.fullStateSize  = config.recurrentSize + self.latentSize + self.smLatentSize
        self.wmFullStateSize = config.recurrentSize + self.latentSize

        self.actor                             = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device,                                  config.actor          ).to(self.device)
        self.critic                            = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        self.encoder                           = EncoderConv(observationShape, self.config.encodedObsSize,                                             config.encoder        ).to(self.device)
        self.decoder                           = DecoderConv(self.wmFullStateSize, observationShape,                                                   config.decoder        ).to(self.device)
        self.recurrentModel                    = RecurrentModel(config.recurrentSize, self.latentSize, actionSize,                                     config.recurrentModel ).to(self.device)
        self.priorNet                          = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        self.posteriorNet                      = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet   ).to(self.device)
        self.rewardPredictor                   = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)

        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                                                                config.continuation   ).to(self.device)

        self.buffer         = ReplayBuffer(observationShape, actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)

        self.totalEpisodes       = 0
        self.totalEnvSteps       = 0
        self.totalGradientSteps  = 0
        self.totalSelfModelSteps = config.batchSize * (config.batchLength - 1)


    def worldModelTraining(self, data, smLatentStates):
        encodedObservations = self.encoder(data.observations.view(-1, *self.observationShape)).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState  = torch.zeros(self.config.batchSize, self.recurrentSize,    device=self.device)
        previousLatentState     = torch.zeros(self.config.batchSize, self.latentSize,       device=self.device)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState    = posterior

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        reconstructionMeans        =  self.decoder(fullStates.view(-1, self.wmFullStateSize)).view(self.config.batchSize, self.config.batchLength-1, *self.observationShape)
        reconstructionDistribution =  Independent(Normal(reconstructionMeans, 1), len(self.observationShape))
        reconstructionLoss         = -reconstructionDistribution.log_prob(data.observations[:, 1:]).mean()

        fullStates = torch.cat((fullStates, smLatentStates), dim=-1)

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss # I think that the reconstruction loss is relatively a bit too high (11k)s
        
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.dones[:, 1:])
            worldModelLoss      += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}

        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def selfModelTraining(self, data):
        totalGradientSteps = self.totalGradientSteps
        config = self.configFile
        scaler = GradScaler()
        sim_real = config.dreamer.selfModel.sim_real
        arm_ee = config.dreamer.selfModel.arm_ee
        seed_num = config.seed
        robotid = config.robotID
        FLAG_PositionalEncoder = config.dreamer.selfModel.positionalEncoder

        # 0:OM, 1:OneOut, 2: OneOut with distance
        different_arch = 0

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
        config = config.dreamer
        if different_arch != 0:
           LOG_PATH += 'diff_out_%d' % different_arch
        os.makedirs(LOG_PATH + "/image/", exist_ok=True)
        os.makedirs(LOG_PATH + "/best_model/", exist_ok=True)

        # Encoders
        """arm dof = 2+3; arm dof=3+3"""
        # Stratified sampling
        perturb = True  # If set, applies noise to sample positions
        inverse_depth = False  # If set, samples points linearly in inverse depth
        # Hierarchical sampling
        n_samples_hierarchical = 64  # Number of samples per ray  # again no use found, (eze)
        perturb_hierarchical = False  # If set, applies noise to sample positions  # again no use found, (eze)

        # Training
        Camera_FOV = 45.
        camera_angle_y = Camera_FOV * np.pi / 180.
        focal = 0.5 * 64 / np.tan(0.5 * camera_angle_y)
        tr = config.selfModel.tr  # training ratio
        batchSize = int(config.batchSize)
        batchLength = int(config.batchLength)
        training_imges_snapshot = data.nextObservations.clone()
        training_angles_snapshot = data.angles.clone()
        height, width = training_imges_snapshot[0, 0].shape[1:]
        training_imges_snapshot = training_imges_snapshot.reshape(batchSize, batchLength, -1, height, width)
        training_angles_snapshot = training_angles_snapshot.reshape(batchSize, batchLength, DOF)
        train_amount = int((batchLength - 1) * tr)
        loss_v_last = np.inf
        patience = 0
        min_loss = np.inf
        rays_o, rays_d = get_rays(int(0.25 * height), int(0.25 * width), focal)
        chunksize = eval(config.selfModel.chunkSize)  # Modify as needed to fit in GPU memory

        # Early Stopping
        warmup_iters = 400  # Number of iterations during warmup phase
        warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
        n_restarts = 1000  # Number of times to restart if training stalls

        record_file_train = open(LOG_PATH + "/log_train.txt", "a")
        record_file_val = open(LOG_PATH + "/log_val.txt", "a")
        Patience_threshold = config.selfModel.patienceThreshold  # original: 100, (eze)

        # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
        # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'
        if totalGradientSteps == 0:
            pretrained_model_pth = LOG_PATH + '/best_model/best_model.pt'
        else:
            pretrained_model_pth = LOG_PATH + '/best_model/model_epoch%d.pt' % (self.totalSelfModelSteps)
            self.totalSelfModelSteps = (totalGradientSteps + 1) * config.batchSize * (config.batchLength - 1)

        for _ in range(n_restarts):

            model, optimizer = init_models(d_input=(DOF - 2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                           d_filter=config.selfModel.d_filter,
                                           output_size=2,
                                           lr=5e-4,  # 5e-4
                                           pretrained_model_pth=pretrained_model_pth,
                                           FLAG_PositionalEncoder=FLAG_PositionalEncoder,
                                           return_output=True
                                           )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
            latents = torch.zeros(batchSize, batchLength - 1, config.selfModel.d_filter // 4,
                                  device=device)  # batchLength -1 because WM ignores first fullstate, (eze)

            for t in range(batchLength - 1):
                one = time.time()
                angles = training_angles_snapshot[:, t]
                imges = training_imges_snapshot[:, t].permute(0, 2, 3, 1).cpu().numpy()  # permute because cv2 uses channel last, (eze)

                # Pick an image as the target. # RGB -> colorfilter -> binary, (eze)
                maskedImg = torch.zeros(config.batchSize, training_imges_snapshot.shape[3], training_imges_snapshot.shape[4], device=device, dtype=torch.float32)
                for j in range(len(imges)):
                    hsvImg = cv2.cvtColor(imges[j], cv2.COLOR_RGB2HSV)
                    lower = np.array([140, 80, 80])
                    upper = np.array([170, 255, 255])
                    mask = cv2.inRange(hsvImg, lower, upper)

                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        largest = max(contours, key=cv2.contourArea)
                        clean_mask = np.zeros_like(mask)
                        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
                    else:
                        clean_mask = np.zeros_like(mask)
                    maskedImg[j] = ((torch.from_numpy(clean_mask).to(device).float() / 255.0) > 0.5)
                target_img = crop_center(maskedImg)  # >0.5 for binary 1 or 0, (eze)
                matplotlib.image.imsave("testMasked.png", maskedImg[0].cpu())
                matplotlib.image.imsave("testReal.png", training_imges_snapshot[0, 0].mean(0).cpu())
                matplotlib.image.imsave("testTarget.png", target_img[0].cpu())
                target_img = target_img.reshape([batchSize, -1])

                if t < train_amount:
                    model.train()

                    # Run one iteration of TinyNeRF and get the rendered RGB image.
                    with autocast("cuda"):
                        outputs, latents[:, t] = model_forward(config, rays_o, rays_d,
                                                               near, far, model,
                                                               chunksize=chunksize,
                                                               arm_angle=angles,
                                                               DOF=DOF,
                                                               output_flag=different_arch,
                                                               n_samples=config.selfModel.nSamples)
                    two = time.time()
                    print(two - one)
                    # Backprop!
                    rgb_predicted = outputs['rgb_map']
                    optimizer.zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_train = loss.item()
                    three = time.time()

                else:
                    # Evaluate testing
                    model.eval()
                    torch.no_grad()
                    valid_psnr = []
                    valid_image = []

                    # Run one iteration of TinyNeRF and get the rendered RGB image.
                    with autocast("cuda"):
                        outputs, latents[:, t] = model_forward(config, rays_o, rays_d,
                                                               near, far, model,
                                                               chunksize=chunksize,
                                                               arm_angle=angles,
                                                               DOF=DOF,
                                                               output_flag=different_arch,
                                                               n_samples=config.selfModel.nSamples)

                    rgb_predicted = outputs['rgb_map']
                    with autocast("cuda"):
                        v_loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
                    np_image = rgb_predicted.reshape(
                        [-1, int(height * 0.25), int(width * 0.25), 1]).detach().cpu().numpy()
                    valid_image.append(np_image[:6])

            loss_valid = np.mean(v_loss.item())

            #print("SM-Loss:", loss_valid, 'patience', patience)
            scheduler.step(loss_valid)

            # save test image
            np_image_combine = np.hstack(valid_image[0])
            np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
            np_image_combine = np.clip(np_image_combine, 0, 1)
            matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
            if Flag_save_image_during_training and totalGradientSteps % 50 == 0:
                matplotlib.image.imsave(
                    LOG_PATH + '/image/' + '%d.png' % (self.totalSelfModelSteps),
                    np_image_combine)

            record_file_train.write(str(loss_train) + "\n")
            record_file_val.write(str(loss_valid) + "\n")
            torch.save(model.state_dict(), LOG_PATH + '/best_model/model_epoch%d.pt' % (self.totalSelfModelSteps))

            if min_loss > loss_valid:
                """record the best image and model"""
                min_loss = loss_valid
                matplotlib.image.imsave(LOG_PATH + '/image/' + 'best.png', np_image_combine)
                torch.save(model.state_dict(), LOG_PATH + '/best_model/best_model.pt')
                patience = 0
                success = True
            elif loss_valid == loss_v_last:
                print("restart")
                success = False
            else:
                patience += 1
                success = True

            loss_v_last = loss_valid
            # os.makedirs(LOG_PATH + "epoch_%d_model" % i, exist_ok=True)
            # torch.save(model.state_dict(), LOG_PATH + 'epoch_%d_model/nerf.pt' % i)
            # torch.cuda.empty_cache()    # to save memory
            latents = latents.reshape(config.batchSize, (config.batchLength - 1), latents.shape[-1])
            if patience > Patience_threshold:
                break
            if success:
                print('SelfModel-Training successful!')
                break

        record_file_train.close()
        record_file_val.close()
        metrics = {
            "smLoss" : v_loss.item()
        }
        return latents, loss_valid, metrics

    def behaviorTraining(self, fullState):
        recurrentState, latentState, smLatentState = torch.split(fullState, (self.recurrentSize, self.latentSize, self.smLatentSize), -1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState, smLatentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)

        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        print(predictedRewards[2])
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)

        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, seed=None, evaluation=False, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(1, self.recurrentSize, device=self.device), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed= (seed + self.totalEpisodes if seed else None))
            encodedObservation = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))
            angles = torch.as_tensor(env.unwrapped.data.qpos.copy()[:self.config.selfModel.dof], device=self.device, dtype=torch.float32).unsqueeze(0)

            currentScore, stepCount, done, frames = 0, 0, False, []
            while not done:
                recurrentState                  = self.recurrentModel(recurrentState, latentState, action)
                latentState, _                  = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))
                with torch.no_grad():
                    smLatentState, smPrediction = selfmodelEvalForward(config=self.configFile, observationShape=self.observationShape, data=angles, envInteraction=True)
                    target_img = crop_center(torch.from_numpy(observation).unsqueeze(0)).mean(dim=-1 if observation.shape[-1] in (1, 3) else 1).to(device).reshape(1, -1)
                    with autocast("cuda"):
                        sm_loss = torch.nn.functional.mse_loss(smPrediction, target_img)
                #print("smLatentStateSize: ", smLatentState.size(), "recurrentStateSize: ", recurrentState.size(), "latentStateSize: ", latentState.size())  # debugging, (eze)

                action          = self.actor(torch.cat((recurrentState, latentState, smLatentState), -1))
                actionNumpy     = action.cpu().numpy().reshape(-1)

                nextObservation, reward, done = env.step(actionNumpy)
                print(reward)
                angles = torch.as_tensor(env.unwrapped.data.qpos.copy()[:self.config.selfModel.dof], device=self.device, dtype=torch.float32)  # qpos from documentation, (eze)
                vel = torch.as_tensor(env.unwrapped.data.qvel.copy()[:self.config.selfModel.dof], device=self.device, dtype=torch.float32)  # only used in Dreams, (eze)
                if not evaluation:
                    self.buffer.add(observation, actionNumpy, reward, nextObservation, done, angles, vel)

                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio warning
                    targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                    frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
                angles = angles.unsqueeze(0)
                observation = nextObservation
                
                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        return sum(scores)/numEpisodes if numEpisodes else None, np.mean(sm_loss.item())
    

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps,
            'totalSelfModelSteps'   : self.totalSelfModelSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        self.totalSelfModelSteps = checkpoint['totalSelfModelSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])

