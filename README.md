[![Run on Replicate](https://replicate.com/zsxkib/step1x-edit/badge)](https://replicate.com/zsxkib/step1x-edit)

# Step1X-Edit: Run advanced image editing locally

This repository provides a Cog container for **Step1X-Edit**, an advanced image editing model developed by StepFun AI. It lets you edit images based on a reference image and a text instruction, using the approach from their research.

**Model links:**
*   Project page: [step1x-edit.github.io](https://step1x-edit.github.io/)
*   Technical report (Arxiv): [arxiv.org/abs/2504.17761](https://arxiv.org/abs/2504.17761)
*   Original model (Hugging Face): [stepfun-ai/Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit)
*   Online demo: [Step1X-Edit Space](https://huggingface.co/spaces/stepfun-ai/Step1X-Edit)
*   This Cog packaging by: [zsxkib on GitHub](https://github.com/zsxkib) / [@zsakib\_ on Twitter](https://twitter.com/zsakib_)

## Prerequisites

*   **Docker**: You'll need Docker to build and run the container. [Install Docker](https://docs.docker.com/get-docker/).
*   **Cog**: You'll need Cog to build and run this model locally. [Install Cog](https://github.com/replicate/cog#install).
*   **NVIDIA GPU**: You need an NVIDIA GPU with enough memory to run the model. Check the original model's documentation for specifics (it might need more than 40 GB).

## Run locally

Cog makes it straightforward to run this model locally. It handles building the container and downloading the model weights automatically.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/zsxkib/cog-Step1X-Edit.git
    cd cog-Step1X-Edit
    ```

2.  **Run the model:**
    The first time you run `cog predict`, it builds the container and downloads the weights (which can be several gigabytes), so it might take a few minutes. Runs after that will be much faster.

    You can provide the input image as a local file path using `@` or as a public URL:

    ```bash
    # Example using a local file path
    cog predict \
      -i image=@path/to/your_image.jpg \
      -i prompt="make the sky look like a beautiful sunset"
    ```

    ```bash
    # Example using a URL
    cog predict \
      -i image=https://raw.githubusercontent.com/replicate/cog/main/docs/logo.png \
      -i prompt="turn the background blue"
    ```

    Cog saves the edited image and prints the path, like `/tmp/step1x_edit_output.webp`.

    **You can change settings too:**
    ```bash
    cog predict \
      -i image=@path/to/another_image.png \
      -i prompt="remove the car from the background" \
      -i size_level=768 \
      -i seed=12345 \
      -i output_format="png"
    ```
    You can change settings by adding more `-i` arguments. Check `predict.py` to see all the options, like `size_level`, `seed`, `output_format`, and `output_quality`. The `negative_prompt` is empty and other settings like guidance and steps are fixed in this setup.

## How it works

Cog uses `cog.yaml` to define the environment and `predict.py` to set up and run the model. When you run it the first time, the `setup` function in `predict.py` downloads the main Step1X-Edit model weights using `pget` if they're not already cached. The Qwen model for understanding the prompt and image is downloaded using the Hugging Face Hub tools.

## License

The original Step1X-Edit model uses the Apache 2.0 license. The code in this repository for packaging the model with Cog uses the MIT license. Please respect the original model's usage restrictions and license terms.

---

⭐ Star this repo on [GitHub](https://github.com/zsxkib/cog-Step1X-Edit)!

👋 Follow me on [Twitter/X](https://x.com/zsakib_)