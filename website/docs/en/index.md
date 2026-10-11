---
template: home.html
title: yonga — run Hugging Face models on Intel AI PCs
description: >-
  yonga inspects your machine, prepares a validated OpenVINO model for the NPU, GPU or CPU,
  and tells you why when something cannot run.
hide:
  - navigation
  - toc
home:
  badge: v0.5.1 · first public release
  badge_link: https://github.com/ooguz/yonga/releases/tag/v0.5.1
  title: Hugging Face models on your Intel <em>NPU, GPU and CPU</em>.
  lead: >-
    Give yonga a model id. It inspects your machine, picks a device and a quantization profile,
    converts the model with OpenVINO, checks that it really runs, and explains every fallback
    instead of dumping a compiler traceback.
  primary: Get started
  primary_link: getting-started/
  secondary: View on GitHub
  install_label: Install
  install_tabs:
    - name: AppImage
      code: |-
        curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
        chmod +x yonga-0.5.1-x86_64.AppImage
        ./yonga-0.5.1-x86_64.AppImage self install
    - name: pip
      code: |-
        python3 -m venv ~/.venvs/yonga
        ~/.venvs/yonga/bin/pip install "yonga[openvino,export,serve]"
  install_note: Linux x86_64 · no sudo · nothing outside your home directory
  terminal_caption: A real run on an Intel Core Ultra 7 268V, trimmed.
  steps_title: Three commands from model id to answer
  steps:
    - command: yonga inspect Qwen/Qwen3-0.6B
      text: See the architecture, size, license and what preparing it on this machine would take, before anything is downloaded.
    - command: yonga prepare Qwen/Qwen3-0.6B
      text: Download only the files the conversion needs, quantize, repair known issues, and compile a smoke test on the target device.
    - command: yonga chat Qwen/Qwen3-0.6B
      text: Chat in the terminal, or run <code>yonga serve</code> for an OpenAI-compatible API your existing tools can talk to.
  features_title: Built for the hardware you already have
  features:
    - icon: material/chip
      title: Device-aware, NPU included
      text: yonga probes the NPU, GPU and CPU, chooses a profile the device is known to handle, and never switches devices behind your back when you name one.
    - icon: material/check-decagram-outline
      title: Validated, reproducible models
      text: Every artifact pins the source revision, records the profile and its evidence, carries file hashes, and passed a real generation test before it is called ready.
    - icon: material/message-alert-outline
      title: Failures you can act on
      text: A data-driven registry recognises compiler, exporter and driver failures and turns them into a diagnostic with a code, a reason and the next command to run.
    - icon: material/api
      title: OpenAI-compatible server
      text: Streaming, stop sequences, tool calling on supported families and images for vision models, served from one local artifact.
    - icon: material/shield-lock-outline
      title: Private by design
      text: No telemetry. yonga also blocks OpenVINO's own usage reporting, talks to the Hugging Face Hub only when it has to, and can run offline from what is already cached.
    - icon: material/package-variant-closed
      title: One file to install
      text: The AppImage bundles Python and OpenVINO. Exports run in managed environments, so your system Python is never touched and no root access is needed.
  proof_title: Measured, not promised
  proof_lead: >-
    Every claim on this site was run on one reference machine: an Intel Core Ultra 7 268V
    (Lunar Lake) with an Arc iGPU and the Intel AI Boost NPU, on Ubuntu 26.04.
  stats:
    - value: "12"
      label: architecture families verified on the reference machine, from Qwen and Llama to Gemma, Granite and Phi
    - value: "~68 tok/s"
      label: Qwen3-0.6B on the NPU, INT4 symmetric, measured with <code>yonga bench</code>
    - value: "1.4 GiB"
      label: downloaded for Qwen3-0.6B, only the files the conversion reads
  proof_link: See the tested hardware and models
  proof_href: hardware/
  pipeline_title: What prepare does, step by step
  pipeline:
    - Inspecting source
    - Resolving compatibility
    - Preflight
    - Downloading
    - Converting
    - Validating / repairing
    - Probing runtime
  pipeline_note: Each step is logged to a job you can read back with <code>yonga logs</code>.
  cta_title: Ready to try it?
  cta_text: Install yonga, run <code>yonga doctor</code>, and prepare your first model in a few minutes.
  cta_primary: Read the quick start
  cta_secondary: Browse the docs
  cta_secondary_link: concepts/
  disclaimer: yonga is an independent open-source project and is not affiliated with or endorsed by Intel. Licensed under the GNU AGPL v3.0 or later.
---
