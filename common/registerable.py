class Registrable(object):
    """
    A class that collects all registered components,
    adapted from `common.registrable.Registrable` from AllenNLP
    """
    registered_components = dict()

    @staticmethod
    def register(name):
        def register_class(cls):
            if name in Registrable.registered_components:
                raise RuntimeError('class %s already registered' % name)

            Registrable.registered_components[name] = cls
            return cls

        return register_class

    @staticmethod
    def by_name(name):
        return Registrable.registered_components[name]
