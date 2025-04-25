# Get the current directory
$CurrentDir = Get-Location

# Run the Docker container with volume mapping
docker run --gpus all -it --rm -v "$($CurrentDir.Path):/workspace" my_llm_arena