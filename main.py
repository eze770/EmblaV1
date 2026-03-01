import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation
import torch
import argparse
from tqdm import tqdm
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
from func import selfmodelEvalForward
import sm_train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch version:", torch.__version__)
print("device: ", device)


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    
    env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=200), render_only=True), (64, 64))))
    envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(AddRenderObservation(gym.make(config.environmentName, render_mode="rgb_array", max_episode_steps=200), render_only=True), (64, 64))))
    
    observationShape, actionSize, actionLow, actionHigh, dt = getEnvProperties(env)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}, dt {dt}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, dt, device, config.dreamer, config)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    selfmodelEvalForward(config=config, observationShape=observationShape, data=torch.as_tensor([0, 0, 0, 0, 0, 0, 0]), initializeLatents=True)  # save one random latent state for Dreamer start, (eze)
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    for _ in tqdm(range(iterationsNum), desc="OverallProgress", colour="green"):
        for _ in tqdm(range(config.replayRatio), desc="Dream", colour="blue"):
            sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            if (config.dreamer.selfModel.nIters / config.dreamer.batchLength * config.dreamer.batchSize) - dreamer.totalGradientSteps >= 0:
                smLatentStates = sm_train.main(config, sampledData, dreamer.totalGradientSteps)  # initialize SelfModel training, (eze)
            else:
                smLatentStates = torch.zeros(config.dreamer.batchSize, config.dreamer.batchLength-1, config.dreamer.selfModel.d_filter // 4, device=device)  # batchLength -1 because WM ignores first fullstate, (eze)
                for t in range(1, config.dreamer.batchLength):
                    for b in range(len(sampledData.angles[:, t])):
                        smLatentStates[b, t] = selfmodelEvalForward(config=config, observationShape=observationShape, data=sampledData.angles[b, t])  # separated training and this step because SM is not dynamic, which means that sampledData would be way less, (eze)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData, smLatentStates)  # initial states also contains SM Latents (used for continuationpredictor), (eze)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates, sampledData.angles, sampledData.vel)
            dreamer.totalGradientSteps += 1

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pusher-v1.yml")
    main(parser.parse_args().config)
