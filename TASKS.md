# Legacy Ghost Training Tasks

This file is no longer the default runtime queue.

Ghost now uses `TASKS.json` as the primary object-backed task queue, and the MCP server can manage that queue through the `list_training_tasks`, `create_training_task`, `update_training_task`, and `delete_training_task` tools.

Keep this file only if you intentionally want a legacy markdown queue and explicitly point the training agent at it.


