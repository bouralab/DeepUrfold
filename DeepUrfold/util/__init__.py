def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2boolorval(v, default=None, choices=None):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        if choices is not None:
            if v in choices:
                return v
            else:
                raise argparse.ArgumentTypeError(f"{v} not in {choices}")
        return default

def str2boolorlist(v, default="raw"):
    if len(v) == 0:
        return [default]
    elif len(v) == 1 and v[0].lower() in ('yes', 'true', 't', 'y', '1'):
        return [default]
    elif len(v) == 1 and v[0].lower() in ('no', 'false', 'f', 'n', '0'):
        return None
    return v
