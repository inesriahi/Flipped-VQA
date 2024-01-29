import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        vqa_loss, vaq_loss, qav_loss = model(data)

        loss = vqa_loss + vaq_loss + qav_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()
        vaq_loss_value = vaq_loss.item()
        qav_loss_value = qav_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(vaq_loss=vaq_loss_value)
        metric_logger.update(qav_loss=qav_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.debug:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.debug:
            print("data:", data)
            print("data['text_id']['vqa'].shape:", data['text_id']['vqa'].shape) # to check why in the model it is taking the options as separate dim because i don't see that in the code itself
        answer = data['answer'].cuda() # contains integer indecies for the correct answers in a batch
        bsz = answer.shape[0] # answer.shape [bsz] answer => tensor([3, 2, 2, 1, 4, 4, 0, 2], device='cuda:0')

        if args.debug:
            print(f"Answer shape: {answer.shape}")
            print(f"Answer: {answer}")

        with torch.no_grad():
            if args.is_generation_task:
                individual_losses, most_similar_indecies = model(data, inference=True) # torch.Size([8, 5, 127]) (bsz, num_options, seq_len-1), most_similar_indecies (bsz, num_options)
            else:
                individual_losses = model(data, inference=True) # torch.Size([8, 5, 127]) (bsz, num_options, seq_len-1)
        
        if args.debug:
            print(f"individual_losses shape: {individual_losses.shape}")
            print(f"individual_losses: {individual_losses}")

        count = (individual_losses != 0).sum(-1) # [8, 5] where each element of the 5 options shows the number of nonzero losses. Why != 0? I think this is because ignore_index=0 so this represents the 
        if args.debug:
            print(f"Count shape: {count.shape}")
            print(f"Count: {count}")

        prediction = (individual_losses.sum(-1) / count).argmin(-1) # [8,] select the index of the option with the least loss
        if args.debug:
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction: {prediction}")

        
        if args.is_generation_task:
            eval_nearest_sentence = (answer == most_similar_indecies)
            acc_nearest_sentence = eval_nearest_sentence.sum().item() / bsz
        
        eval_exact_match = (answer == prediction)
        acc_exact_match = eval_exact_match.sum().item() / bsz
        if args.debug:
            print(f"Eval_exact_match: {eval_exact_match}")
            print(f"Accuracy: {acc_exact_match}")

        misc.log_qtype(data, eval_exact_match, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if args.is_generation_task:
            metric_logger.update(n=bsz, acc_exact_match=acc_exact_match, acc_nearest_sentence=acc_nearest_sentence)
        else:
            metric_logger.update(n=bsz, acc_exact_match=acc_exact_match)
        if args.debug:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if args.debug:
        sys.exit(1)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

