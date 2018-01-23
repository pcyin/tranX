# coding=utf-8


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        # for GenToken actions only
        self.copy_from_src = False
        self.src_token_position = -1
