import os
import torch
import csv
import hydra

from rlbench.backend import task as rlbench_task
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Value

from VIHE.models.utils.mvt import MVT
from VIHE.models.utils.mvtpp import MVTPP
from VIHE.models.utils.utils import load_agent
from VIHE.models.rvt_agent import RVTAgent
from VIHE.models.rvtpp_agent import RVTPPAgent
from VIHE.utils import get_logdir, create_obs_config, get_tasks, CustomMultiTaskRLBenchEnv
from VIHE.utils import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning
from VIHE.logger.tensorboardmanager import TensorboardManager

from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.backend.utils import task_file_to_task_class

from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator


@torch.no_grad()
def eval(cfg, agent):
    cfg.log_dir = get_logdir(cfg)

    agent.eval()
    agent.load_clip()
    camera_resolution = [cfg.agent.img_size, cfg.agent.img_size]
    obs_config = create_obs_config(
        cfg.dataset.cameras, camera_resolution, method_name="")
    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []

    tasks = get_tasks(cfg.tasks)
    print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=cfg.eval.datafolder,
        episode_length=cfg.eval.episode_length,
        headless=cfg.eval.headless,
        swap_task_every=cfg.eval.episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if cfg.eval.save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{cfg.device[0]}"

    # create metric saving writer
    csv_file = f"{cfg.eval.log_name}.csv"
    if not os.path.exists(os.path.join(cfg.log_dir, csv_file)):
        with open(os.path.join(cfg.log_dir, csv_file), "w") as csv_fp:
            fieldnames = ["task", "success rate",
                          "length", "total_transitions"]
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        for ep in range(cfg.eval.episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=cfg.eval.episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=cfg.eval.replay_ground_truth,
            )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            print(
                f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
            )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]

        with open(os.path.join(cfg.log_dir, csv_file), "a") as csv_fp:
            fieldnames = ["task", "success rate",
                          "length", "total_transitions"]
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_results = {"task": task_name}
            for s in summaries:
                if s.name == "eval_envs/return":
                    csv_results["success rate"] = s.value
                elif s.name == "eval_envs/length":
                    csv_results["length"] = s.value
                elif s.name == "eval_envs/total_transitions":
                    csv_results["total_transitions"] = s.value
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)
            csv_writer.writerow(csv_results)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(
            f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

    eval_env.shutdown()
    csv_fp.close()

    agent.unload_clip()
    return scores


def experiment(cfg):
    print("Config:\n{}".format(OmegaConf.to_yaml(cfg)))
    device = f"cuda:{cfg.device[0]}"
    tb = TensorboardManager(cfg.log_dir)

    if cfg.agent.name == "rvt":
        rvt = MVT(cfg.agent, renderer_device=device).to(device)
        agent = RVTAgent(
            cfg.agent,
            network=rvt,
            cameras=cfg.dataset.cameras,
            scene_bounds=cfg.dataset.scene_bounds
        )

    elif cfg.agent.name == "vihe":
        vihe = MVTPP(cfg.agent, renderer_device=device).to(device)
        agent = RVTPPAgent(
            cfg.agent,
            network=vihe,
            cameras=cfg.dataset.cameras,
            scene_bounds=cfg.dataset.scene_bounds
        )

    agent.build(training=False, device=device)
    load_agent(cfg.agent.model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)

    scores = eval(cfg, agent)
    print(f"model {cfg.agent.model_path}, scores {scores}")
    task_scores = {}
    for i in range(len(cfg.tasks)):
        task_scores[cfg.tasks[i]] = scores[i]

    print("save ", task_scores)
    tb.update("eval", 0, task_scores)
    tb.writer.flush()
    tb.close()


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    experiment(cfg)


if __name__ == "__main__":
    main()
