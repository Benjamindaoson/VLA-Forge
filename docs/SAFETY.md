# Research software safety notice

VLA-Forge is not a safety-certified robot controller and is not validated for use on actual machines.

- Hardware/firmware must enforce independent emergency stop, joint limits, workspace limits, collision monitoring and supervisor watchdogs.
- ActionChunkScheduler only validates timing, dimensions and numerical bounds. It does NOT guarantee collision-free or human-safe motion.
- Treat recommended actions as untrusted and requiring independent verification.
- Keep the development API on loopback; it has no authentication, tenant authorization, rate limits or secure telemetry storage.
- Before physical rollout, implement simulation -> shadow -> human approval -> independent safety validation stages.
- Logs distinguish predicted actions, commanded actions and observed actuator motion.
- Customer videos, robot state and training artifacts are not shareable across customers without affirmative permission.
