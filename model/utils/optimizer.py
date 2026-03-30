import math
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer_scheduler(parameters, num_samples, **kwargs):
    """
    - linear           : Warmup 후에는 LR = 1.0(고정)
    - cosine           : Warmup 후 Cosine Decay
    - cosine_restart   : Warmup 후 Cosine Decay with Restart
    """    
    type = kwargs['optimizer']
    max_lr = kwargs['max_lr']
    min_lr = kwargs['min_lr']
    lr_decay_rate = kwargs['lr_decay_rate']
    weight_decay = kwargs['weight_decay']
    batch_size = kwargs['batch_size']
    accumulation_steps = kwargs['accumulation_steps']
    epochs = kwargs['epochs']
    warmup_epochs = kwargs['warmup_epochs']
    scheduler = kwargs['scheduler']
    t_0 = kwargs['restart_period']
    world_size = kwargs['word_size']
    
    # debugging
    assert 0 < lr_decay_rate <= 1, "lr_decay_rate should be between 0 and 1 for decay"

    if type=='adam':
        optimizer = optim.Adam(parameters, lr=max_lr, weight_decay=weight_decay)
    elif type=='adamw':
        optimizer = optim.AdamW(parameters, lr=max_lr, weight_decay=weight_decay)
    elif type=='sgd':
        optimizer = optim.SGD(parameters, lr=max_lr, weight_decay=weight_decay)
        
    global_batch_size = batch_size*world_size
    total_batches = math.ceil(num_samples / global_batch_size)
    effective_batches = total_batches // accumulation_steps
    total_steps = effective_batches * epochs
    warmup_steps = warmup_epochs * effective_batches

    def lr_lambda_linear(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            return 1.0

    def lr_lambda_cosine(current_step):
        min_lr_ratio = min_lr / max_lr
             
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)

    def lr_lambda_cosine_restart(current_step):
        min_lr_ratio = min_lr / max_lr
        restart_steps = t_0 * effective_batches  # 한 주기의 총 스텝 수 (warmup_steps 포함)

        # 현재 주기와 주기 내 스텝 계산
        cycle = current_step // restart_steps  # 현재 몇 번째 주기인지
        step_in_cycle = current_step % restart_steps  # 주기 내에서의 스텝

        # 주기별 최대 학습률 스케일링
        cycle_max_lr_scale = (lr_decay_rate ** cycle) if cycle > 0 else 1.0

        # Warmup 단계 체크
        if warmup_steps > 0 and step_in_cycle < warmup_steps:
            # 주기 내 warmup 구간
            return float(step_in_cycle) / float(warmup_steps) * cycle_max_lr_scale
        else:
            # Warmup 이후 코사인 디케이 구간
            step_in_decay = step_in_cycle - warmup_steps if warmup_steps > 0 else step_in_cycle

            # 코사인 디케이는 warmup 이후 구간만 고려
            decay_steps = restart_steps - warmup_steps if warmup_steps > 0 else restart_steps
            progress_in_decay = float(step_in_decay) / float(decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_in_decay))
            lr_scaled = (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay) * cycle_max_lr_scale
            return max(min_lr_ratio, lr_scaled)

    if scheduler == 'linear':
        lr_func = lr_lambda_linear
        print("Using Linear scheduler")
    elif scheduler == 'cosine':
        lr_func = lr_lambda_cosine
        print("Using Cosine decay scheduler")
    elif scheduler == 'cosine_restart':
        lr_func = lr_lambda_cosine_restart
        print("Using Cosine decay with Restart scheduler")
    else:
        raise NotImplementedError("Not Implemented Scheduler")
    
    scheduler = {
        'scheduler': LambdaLR(optimizer, lr_func),
        'interval': 'step',
        'frequency': 1,
    }

    return optimizer, scheduler
