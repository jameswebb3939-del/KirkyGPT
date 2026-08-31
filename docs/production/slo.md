# EC Pro Production Service Level Objectives

## 1. Purpose

This document defines the production Service Level Objectives
(SLOs) for the EC Pro inference service.

The SLOs apply to the stateless inference tier exposed through
the `/chat` endpoint.

They are intended to provide measurable targets for:

- availability;
- inference latency;
- error rate;
- inference queue pressure;
- runtime availability;
- autoscaling behaviour.

The objectives are measured using Prometheus metrics exported by
the EC Pro inference service.

---

## 2. Service Boundary

The production inference request path is:

```text
Client
  ↓
Ingress / external load balancer
  ↓
Kubernetes Service
  ↓
EC Pro inference pod
  ↓
NativeLLMRuntime
  ↓
C++ BatchScheduler
  ↓
llama.cpp
  ↓
GPU / CPU inference backend