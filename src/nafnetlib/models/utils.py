from torch import nn

from .misc import AvgPool2d


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


def dynamic_instantiation(cls_type, opt):
    from . import nafnet

    if hasattr(nafnet, cls_type):
        cls_ = getattr(nafnet, cls_type)
        return cls_(**opt)
    raise ValueError(f"Cannot find model {cls_type}")


def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(network_type, opt)
    return net
