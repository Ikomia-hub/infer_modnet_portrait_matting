<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_modnet_portrait_matting/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_modnet_portrait_matting</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_modnet_portrait_matting">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_modnet_portrait_matting">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_modnet_portrait_matting/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_modnet_portrait_matting.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference with MODNet a trimap-free portrait matting in real time.

![Face restoration codeformer](https://raw.githubusercontent.com/ZHKKKe/MODNet/develop/doc/gif/image_matting_demo.gif)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API 

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
algo = wf.add_task(name="infer_modnet_portrait_matting", auto_connect=True)

# Run workflow on the image
wf.run_on(url="https://images.pexels.com/photos/12508998/pexels-photo-12508998.jpeg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **input_size** (int) - default: '1024': Size of the input image (stride of 32)
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
algo = wf.add_task(name="infer_modnet_portrait_matting", auto_connect=True)

# Set process parameters
algo.set_parameters({
    "input_size" : "1024",
    "cuda" : "True"})

# Run workflow on the image
wf.run_on(url="https://images.pexels.com/photos/12508998/pexels-photo-12508998.jpeg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_modnet_portrait_matting", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/12508998/pexels-photo-12508998.jpeg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

