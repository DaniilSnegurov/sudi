# Локальный сервер моделей (Ollama, портативная сборка в tools\ollama). Слушает только 127.0.0.1.
# Модель скачивается один раз: tools\ollama\ollama.exe pull qwen3.5:9b
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_KEEP_ALIVE = "30m"
$env:OLLAMA_NUM_PARALLEL = "1"
& "$PSScriptRoot\tools\ollama\ollama.exe" serve
