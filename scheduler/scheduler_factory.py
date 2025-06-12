# %%writefile /content/ALBEF/scheduler/scheduler_factory.py
# File: scheduler/scheduler_factory.py

""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler
from .plateau_lr import PlateauLRScheduler


def create_scheduler(args, optimizer):
    # Thay đổi tất cả các truy cập args['key'] thành args.get('key', default_value)
    # Cung cấp giá trị mặc định phù hợp cho từng key
    num_epochs = args.get('epochs', 10) # Default: 10 epochs

    if args.get('lr_noise', None) is not None:
        lr_noise = args.get('lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if args.get('sched') == 'cosine': # Sử dụng .get()
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=args.get('lr_cycle_mul', 1.),
            lr_min=args.get('min_lr', 0.), # Default: 0
            decay_rate=args.get('decay_rate', 1.), # Default: 1
            warmup_lr_init=args.get('warmup_lr', 0.), # Default: 0
            warmup_t=args.get('warmup_epochs', 0), # THAY ĐỔI TRỌNG TÂM Ở ĐÂY, Default: 0
            cycle_limit=args.get('lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=args.get('lr_noise_pct', 0.67),
            noise_std=args.get('lr_noise_std', 1.),
            noise_seed=args.get('seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.get('cooldown_epochs', 0) # Default: 0
    elif args.get('sched') == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=args.get('lr_cycle_mul', 1.),
            lr_min=args.get('min_lr', 0.),
            warmup_lr_init=args.get('warmup_lr', 0.),
            warmup_t=args.get('warmup_epochs', 0),
            cycle_limit=args.get('lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=args.get('lr_noise_pct', 0.67),
            noise_std=args.get('lr_noise_std', 1.),
            noise_seed=args.get('seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.get('cooldown_epochs', 0)
    elif args.get('sched') == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.get('decay_epochs', num_epochs), # Default: num_epochs
            decay_rate=args.get('decay_rate', 1.),
            warmup_lr_init=args.get('warmup_lr', 0.),
            warmup_t=args.get('warmup_epochs', 0),
            noise_range_t=noise_range,
            noise_pct=args.get('lr_noise_pct', 0.67),
            noise_std=args.get('lr_noise_std', 1.),
            noise_seed=args.get('seed', 42),
        )
    elif args.get('sched') == 'plateau':
        mode = 'min' if 'loss' in args.get('eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.get('decay_rate', 1.),
            patience_t=args.get('patience_epochs', 10), # Default: 10
            lr_min=args.get('min_lr', 0.),
            mode=mode,
            warmup_lr_init=args.get('warmup_lr', 0.),
            warmup_t=args.get('warmup_epochs', 0),
            cooldown_t=args.get('cooldown_epochs', 0), # Default: 0
            noise_range_t=noise_range,
            noise_pct=args.get('lr_noise_pct', 0.67),
            noise_std=args.get('lr_noise_std', 1.),
            noise_seed=args.get('seed', 42),
        )

    return lr_scheduler, num_epochs