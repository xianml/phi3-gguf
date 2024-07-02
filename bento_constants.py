CONSTANT_YAML = """
  alias:
    - q4
  project: phi3-gguf
  service_config:
    name: phi3
    traffic:
      timeout: 300
  engine_config:
    model: microsoft/Phi-3-mini-4k-instruct-gguf
    max_model_len: 1024
  chat_template: phi-3
"""
