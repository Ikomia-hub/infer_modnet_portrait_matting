# infer_modnet_portrait_matting


## :rocket: Run with Ikomia API

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()    

# Add the p3m process to the workflow
det = wf.add_task(name="infer_modnet_portrait_matting", auto_connect=True)

# Set process parameters
det.set_parameters({
    "input_size" : "1024", # Select a stride of 32
    "cuda" : "True"})

wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(det.get_input(0))
display(det.get_output(1))

```


## :black_nib: Citation

[Code source](https://github.com/ZHKKKe/MODNet/tree/master) 

```bibtex
@InProceedings{MODNet,
  author = {Zhanghan Ke and Jiayu Sun and Kaican Li and Qiong Yan and Rynson W.H. Lau},
  title = {MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition},
  booktitle = {AAAI},
  year = {2022},
}
```