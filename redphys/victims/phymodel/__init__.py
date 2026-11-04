from ..server import Victim


class PhyModel(Victim):

    def run(self, messages, images, **params):
        return self.chat(messages, **params)
