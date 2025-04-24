# Get the current directory
$CurrentDir = Get-Location

# Run the Docker container with volume mapping
docker run -it --rm -v "$($CurrentDir.Path):/workspace" ltroin/llm_arena