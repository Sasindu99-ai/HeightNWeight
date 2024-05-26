Certainly! Here’s an updated version of your markdown file that includes instructions to ensure `conda` and `poetry` are installed:

# HeightNWeight

---

##### Description
Predict gender by taking weight and height as inputs.

## Setup

Ensure you have `conda` and `poetry` installed.

If you don't have a conda environment set up with Python >= 3.9, you can create one using the following command, where `ENV_NAME` is `tensor`:

```sh
conda create -n tensor python>=3.9
```

Activate the environment:

```sh
conda activate tensor
```

Use `poetry` to install dependencies:

```sh
poetry install
```
