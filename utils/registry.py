class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name):
        def register_fn(module):
            assert module_name not in self
            self[module_name] = module
            return module

        return register_fn


MODELS = Registry()
ENCODERS = Registry()
GENERATORS = Registry()
MAPPINGS = Registry()
DISCRIMINATORS = Registry()
ECOG_ENCODER = Registry()
