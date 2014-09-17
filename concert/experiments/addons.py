"""Add-ons for experiments are standalone extensions which can be attached to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""


class Addon(object):

    """A base addon class."""

    def register(self):
        """Register adds the addon to an experiment. This means all the necessary operations which
        provide the addon functionality should be implemented in this method. This mostly means
        attaching consumers to acquisitions. The method is called by
        :class:`~concert.experiments.base.Experiment.attach`.
        """
        pass

    def unregister(self):
        """Unregister removes the addon from an experiment. This means all the necessary operations
        which provide the addon functionality should be undone by this method. This mostly means
        removing consumers from acquisitions. The method is called by
        :class:`~concert.experiments.base.Experiment.detach`.  """
        pass


class AddonError(Exception):

    """Addon errors."""

    pass
