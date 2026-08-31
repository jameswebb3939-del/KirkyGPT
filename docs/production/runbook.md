
---

# 3. Create `docs/production/runbook.md`

Use:

```markdown
# EC Pro Production Runbook

## 1. Purpose

This runbook defines the operational response procedures for the
EC Pro production inference tier.

Architecture:

```text
Client
  ↓
Ingress
  ↓
Kubernetes Service
  ↓
Inference replicas
  ↓
NativeLLMRuntime
  ↓
C++ BatchScheduler
  ↓
llama.cpp
  ↓
GPU