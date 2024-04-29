# Modified code from https://github.com/Tsedao/MultiRM/blob/master/Scripts/util_ig.py by Song et al.
import torch
from torch import nn
import time
import gc
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from typing import Callable


def random_baseline_integrated_gradients(inputs: torch.Tensor, model: torch.nn.Module, index: int, steps: int, num_random_trials: int, cuda=True, batch=None) -> np.ndarray:
    """

    param: inputs: torch.Tensor: input tensor
    param: model: torch.nn.Module: model
    param: index: int: index of the target label (0 (control) or 1 (case))
    param: steps: int: Riemann approx for IG integral
    param: num_random_trials: int: number of random trials 
    param: cuda: bool: use cuda or not
    param: batch: int: batch size for internal batch size for captum IG

    return: np.ndarray: integrated gradients
    Can't process all due to memory and it is also not necessary to get the motif from low importance sequences
    === NOTES: EXPECTED ROUNDING DIFFERENT FOR SMALL NUMBER. PLEASE USE GPU FOR PERFORMANCE WISE ===

    EXAMPLE:
    ------

    gpu = random_baseline_integrated_gradients(inputs=seq_fasta_one_hot[1:10,:,:], model=model, index=0, steps=50, num_random_trials=10)
    cpu = random_baseline_integrated_gradients(inputs=seq_fasta_one_hot[1:10,:,:], model=model, index=0, steps=50, num_random_trials=10, cuda=False)
    print(gpu.shape)
    print(cpu.shape)
    # Checking gpu and cpu
    np.isclose(gpu, cpu, rtol=1e-03, atol=1e-03, equal_nan=False)
    np.where(~np.isclose(gpu, cpu))
    print(cpu[np.where(~np.isclose(gpu, cpu))])
    print(gpu[np.where(~np.isclose(gpu, cpu))])
    """

    try:
        all_intgrads = []
        length = inputs.shape[-1]        # input shape [1,4,length]
        baseline = torch.FloatTensor(np.zeros(inputs.shape))
        model.eval()
        if cuda:
            # To enable backward for RNN in pytorch where device is GPU or CUDA. Known issue here: https://github.com/pytorch/captum/issues/564
            torch.backends.cudnn.enabled = False
            # Run dummy forward first
            _ = model(torch.rand((1, 4, 1001)))
            # Hard coded device ids. should be changed to more general
            # model = nn.DataParallel(model.cuda(), device_ids=[0,1,2,3])
            # use all GPUs
            model = nn.DataParallel(model.cuda())
            baseline = baseline.to('cuda')
            inputs = inputs.to('cuda')
        else:
            model.cpu()
        for i in range(num_random_trials):
            ig = IntegratedGradients(model)
            integrated_grad = ig.attribute(
                inputs=inputs, internal_batch_size=batch, baselines=baseline, target=index, n_steps=steps)
            all_intgrads.append(integrated_grad.detach().cpu()
                                if cuda else integrated_grad.detach())
            # print('the trial number is: {}'.format(i))
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    finally:
        if cuda:
            torch.cuda.empty_cache()
            model.cpu()
            del baseline
            del inputs
            gc.collect()
    return avg_intgrads


def calculate_outputs_and_gradients(inputs: list[torch.Tensor], model, index: int = 0, device: str = "cpu"):
    """
    By default get the gradient of the control label (0)
    """
    model.eval()
    model.to(device)

    if device != "cpu":
        # To enable backward for RNN in pytorch where device is GPU or CUDA. Known issue here: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled = False

    gradients = []

    start = time.time()
    for i, input_tensor in enumerate(inputs):
        try:
            input = input_tensor
            print(input.shape)
            if i % 100 == 0:
                print(f"Processing sequence {i} to {i+99}")
            input = input.to(device)
            input.requires_grad = True
            input.retain_grad()
            output = model(input)
            output = output.squeeze(0)
            if i % 100 == 0:
                print(f"Output of {i}: {output}, shape {output.shape}")
            # clear grad
            model.zero_grad()
            output[index].backward(retain_graph=True)
            # print(input.grad.detach().cpu().numpy())
            gradient = input.grad.detach().cpu().numpy()
            gradients.append(gradient[0])
            if i == 100:
                return np.array(gradients)
        except Exception as e:  # Free memory
            print(e)
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Time taken: {time.time()-start} seconds")
    # gradients = np.array(gradients)
    # return gradients, target_label_idx


def integrated_gradients(inputs: torch.Tensor, model: torch.nn.Module, predict_and_gradients: Callable, baseline: None | torch.Tensor = None, index: int = 0, steps: int = 50, device: str = "cpu") -> np.ndarray:
    """
    Integrated gradient implementation modified from https://github.com/TianhongDai/integrated-gradient-pytorch/blob/master/main.py 
    """
    if baseline is None:
        baseline = 0 * inputs
    # Scale baseline into inputs linearly for path integral based on steps (m)
    scaled_inputs = [baseline + (float(i) / steps) *
                     (inputs - baseline) for i in range(0, steps + 1)]
    # Calculate the gradients at each point along the path
    # Iterate for each scaled inputs

    grads = predict_and_gradients(
        inputs=scaled_inputs, model=model, index=index, device=device)
    print(len(scaled_inputs))
    # averaging except gradient from the inputs (last one)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.expand_dims(avg_grads, axis=0)
    inputs = inputs.cpu().numpy()
    baseline = baseline.cpu().numpy()
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad


def calculate_outputs_and_gradients(inputs: list[torch.Tensor], model, index: int = 0, device: str = "cpu"):
    """
    By default get the gradient of the control label (0)
    """
    model.eval()
    model.to(device)

    if device != "cpu":
        # To enable backward for RNN in pytorch where device is GPU or CUDA. Known issue here: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled = False

    gradients = []
    start = time.time()
    n_steps = len(inputs)
    for step in range(n_steps):
        input_step = inputs[step]
        step_gradients = []
        for i in range(0, len(input_step)):  # running in batch
            input = input_step[i]
            input.unsqueeze_(0)
            if i % 100 == 0:
                print(f"Processing sequence {i} to {i+99}")
            input = input.to(device)
            input.requires_grad = True
            input.retain_grad()
            output = model(input)
            output = output.squeeze(0)
            if i % 100 == 0:
                print(f"Output shape {output.shape}")
            # clear grad
            model.zero_grad()
            output[index].backward(retain_graph=True)
            # print(input.grad.detach().cpu().numpy())
            gradient = input.grad.detach().cpu().numpy()
            step_gradients.append(gradient[0])
        gradients.append(step_gradients)

    print(f"Time taken: {time.time()-start} seconds")
    gradients = np.array(gradients)
    return gradients


def random_baseline_integrated_gradients(inputs, model, predict_and_gradients, index, steps, num_random_trials, cuda):
    all_intgrads = []
    length = inputs.shape[-1]        # input shape [1,4,length]
    mid = length // 2
    baseline = torch.cuda.FloatTensor(np.zeros(inputs.shape))
    # baseline[:,:,mid] = inputs[:,:,mid]
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, predict_and_gradients,
                                               baseline=baseline,
                                               index=index, steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        # print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads


def calculate_outputs_and_gradients_ori(inputs: list[torch.Tensor], model, index: int = 0, device: str = "cpu"):
    """
    By default get the gradient of the control label (0)
    """
    model.eval()
    model.to(device)

    if device != "cpu":
        # To enable backward for RNN in pytorch where device is GPU or CUDA. Known issue here: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled = False

    gradients = []
    start = time.time()
    for i in range(len(inputs)):
        input = inputs[i]
        # input.unsqueeze_(0)
        if i % 100 == 0:
            print(f"Processing sequence {i} to {i+99}")
            print(f"input shape: {input.shape}")
        input = input.to(device)
        input.requires_grad = True
        input.retain_grad()
        output = model(input)
        output = output.squeeze(0)
        if i % 100 == 0:
            print(f"Output of {i}: {output}, shape {output.shape}")
        # clear grad
        model.zero_grad()
        output[index].backward(torch.ones_like(
            output[index]), retain_graph=True)
        # print(input.grad.detach().cpu().numpy())
        gradient = input.grad.detach().cpu().numpy()
        gradients.append(gradient[0])
        del input
        del output
        gc.collect()

    print(f"Time taken: {time.time()-start} seconds")
    gradients = np.array(gradients)
    return gradients


def calculate_outputs_and_gradients3(inputs: list[torch.Tensor], model, index: int = 0, device: str = "cpu"):
    """
    By default get the gradient of the control label (0)
    """
    model.eval()
    model.to(device)

    if device != "cpu":
        # To enable backward for RNN in pytorch where device is GPU or CUDA. Known issue here: https://github.com/pytorch/captum/issues/564
        torch.backends.cudnn.enabled = False

    gradients = []
    start = time.time()
    n_steps = len(inputs)
    for step in range(n_steps):
        input_step = inputs[step]
        step_gradients = np.array([], dtype="float32")
        batch = 2
        for i in range(0, len(input_step), batch):  # running in batch
            input = input_step[i:i+batch]
            if i % 100 == 0:
                print(f"Processing sequence {i} to {i+99}")
            print(f"Input shape: {input.shape}")
            input = input.to(device)
            input.requires_grad = True
            input.retain_grad()
            output = model(input)
            output = output.squeeze(0)
            # if i % 100 == 0:
            print(f"Output shape {output.shape}")
            # clear grad
            model.zero_grad()
            output[:, 0].backward(torch.ones_like(
                output[:, 0]), retain_graph=True)
            # print(input.grad.detach().cpu().numpy())
            gradient = input.grad.detach().cpu().numpy()
            print(gradient.shape)
            step_gradients = np.vstack(
                [step_gradients, gradient]) if step_gradients.size else gradient
        gradients.append(step_gradients)

    print(f"Time taken: {time.time()-start} seconds")
    gradients = np.array(gradients)
    return gradients
