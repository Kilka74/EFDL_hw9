# Task1 Report

In this directory I implemented all methods from the first task. In `model.py` I implemented ResNet model. If `pretrained=True` we have model with weights from pytorch. I also fixed first convolution for the training on $32 \times 32$ images and deleted max pool. In case of `pretrained=False` we will have untrained version of model with fixed `layer3`. In `train.py` I have all the training pipeline. `train` is basic train without any distillation, `train_distilled` is the variant with distillation. In my code I have flag `add_mse` in `train_distilled` which corresponds to whether we add mse term in loss or not. I set all the weights practically randomly, may be these parameters need to be tuned. But I tried several proportions and these proportions gave me best results. I logged everything in wandb, here is the [link](https://wandb.ai/kilka74/EFDL_hw9/table?nw=nwuserkilka74)

In this link I have 4 shown runs, every run corresponds to one subtask. We can see that all this methods perform sligtly simliar and not veru stable in case of using standart Adam optimizer. I set the number of epochs equal to 10 or 15 in all cases because I haven't got enough resources and usually it was enough for convergence. So, I decided to use the distilled version of resnet which was trained with activation in loss. We will use it in the next task. All runs I have in `hw9.ipynb`, but due to datasphere bug not all the runs are visible((( I couldn't fix it.

# Task2 Report

I tried to adapt the code to work, but I faced to some strnge mistake while using `torch.fx.symbolic_trace`. I didn't fix it unfortunately so my run was failed after 50 epochs, test accuracy you can see in output.txt inside the `task1` directory. I hope that this task will not be marked as zero points.

In folder `task1` I have all my code and pdf of the notebook.