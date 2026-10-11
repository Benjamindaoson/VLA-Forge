"""Decode recorded MuJoCo qpos checkpoints into named free-body object poses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def free_joint_pose_records(joint_names, joint_types, qpos_addresses, qpos, body_names, body_xpos, body_xquat):
    bodies = {name: idx for idx, name in enumerate(body_names)}
    result = []
    for jid, joint_name in enumerate(joint_names):
        if int(joint_types[jid]) != 0 or joint_name.startswith(("robot0_", "gripper0_")):
            continue
        root = joint_name.removesuffix("_joint0")
        candidates = (f"{root}_main", root)
        body_name = next((name for name in candidates if name in bodies), None)
        address = int(qpos_addresses[jid])
        record = {
            "joint_name": joint_name,
            "body_name": body_name,
            "qpos_xyz_quat_wxyz": np.asarray(qpos[address : address + 7]).tolist(),
        }
        if body_name is not None:
            bid = bodies[body_name]
            record["world_position_xyz"] = np.asarray(body_xpos[bid]).tolist()
            record["world_orientation_wxyz"] = np.asarray(body_xquat[bid]).tolist()
        else:
            record["body_pose_resolved"] = False
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, run_dir = args.root.resolve(), args.run_dir.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark
    from run_m03_interventions import _sim

    from robot_vla.m02 import M02_SUITES
    from robot_vla.preflight import create_libero_env

    protocol = json.loads((root / "configs/data_protocol_v1.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    benchmark_dict = benchmark.get_benchmark_dict()
    suites = {
        alias: benchmark_dict[name](task_order_index=protocol["task_order_index"])
        for alias, name in M02_SUITES.items()
    }
    output_rows = []
    for group in manifest["groups"]:
        group_id = group["group_id"]
        env = create_libero_env(
            LiberoEnv,
            suites[group["source_suite"]],
            int(group["source_task_id"]),
            M02_SUITES[group["source_suite"]],
            protocol,
            int(group["init_state_id"]),
            256,
            int(protocol["evaluation"]["horizons"][M02_SUITES[group["source_suite"]]]),
        )
        try:
            env.reset(seed=int(group["seed"]))
            sim = _sim(env)
            treatment = next(k for k in group["condition_results"] if k not in ("baseline", "sham"))
            for condition in ("baseline", "sham", treatment):
                path = run_dir / "episodes" / f"{group_id}_{condition}.npz"
                with np.load(path, allow_pickle=False) as episode:
                    qpos, qvel, sim_time = episode["sim_qpos"][0], episode["sim_qvel"][0], float(episode["sim_time"][0])
                if len(qpos) != len(sim.data.qpos) or len(qvel) != len(sim.data.qvel):
                    raise ValueError(f"checkpoint/model dimension mismatch for {group_id}/{condition}")
                sim.data.qpos[:] = qpos
                sim.data.qvel[:] = qvel
                sim.data.time = sim_time
                sim.forward()
                if not np.array_equal(np.asarray(sim.data.qpos), qpos):
                    raise ValueError(f"qpos changed while restoring {group_id}/{condition}")
                poses = free_joint_pose_records(
                    sim.model.joint_names,
                    sim.model.jnt_type,
                    sim.model.jnt_qposadr,
                    qpos,
                    sim.model.body_names,
                    sim.data.body_xpos,
                    sim.data.body_xquat,
                )
                output_rows.append({
                    "group_id": group_id,
                    "condition": condition,
                    "checkpoint_qpos_exact": True,
                    "checkpoint_qvel_exact": True,
                    "sim_time": sim_time,
                    "free_joint_objects": poses,
                    "all_free_joint_bodies_resolved": all(p["body_name"] is not None for p in poses),
                })
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"pose_records": len(output_rows), "resolved": sum(r["all_free_joint_bodies_resolved"] for r in output_rows), "total": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
