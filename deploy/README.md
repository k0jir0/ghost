# Ghost Deployment Notes

This folder defines the environment separation expected by the new production-stack layers.

- `dev/` is intended for local experimentation and single-user runs.
- `staging/` is intended for pre-production validation of registry promotions and serving changes.
- `production/` is intended for approved models only, with rollback performed by demoting the current production registry record and re-pointing traffic to the previous approved version.

Operational expectations:

- Keep data, model, and metadata roots isolated per environment.
- Run health checks against the serving surface before promotion.
- Preserve the previous production registry version so rollback is a registry-stage change, not a file recovery exercise.
- Record release state through `ghost.deployment.DeploymentManager` so activations, deactivations, and rollbacks are auditable.
- Connect `ghost.secrets.SecretResolver` to the target environment's vault or workload-identity provider before exposing production endpoints.
