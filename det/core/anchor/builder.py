from mtcv.utils import Registry,build_from_cfg

ANCHORE_GENERATORS = Registry('Anchor generator')

def build_anchor_generator(cfg,default_args=None):
    return build_from_cfg(cfg,ANCHORE_GENERATORS,default_args)