

def mesh(controllers, step, callback=None):
    """
    Scan parameter space of all controllers in the controllers list.
    """

    def drange(start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step

    def scan(controllers, remaining, step, callback):
        if not remaining:
            if callback:
                callback(controllers)
        else:
            controller = remaining[0]

            for param in controller.params.values():
                valid = param.range

                for value in drange(valid[0], valid[1], step):
                    param.value = value
                    scan(controllers, remaining[1:], step, callback)

    scan(controllers, controllers, step, callback)
