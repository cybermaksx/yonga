"""Application services.

Each service orchestrates domain policy and adapter mechanism for one user-facing
operation. Services accept injected adapters and a progress sink; they never
import Rich, OpenVINO, Hugging Face or ``subprocess`` directly.
"""
