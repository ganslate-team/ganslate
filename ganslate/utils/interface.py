import sys
from ganslate.engines.utils import init_engine


def run_interface():
    Interface().run()


class Interface:
    def __init__(self):

        self.COMMANDS = {
            'help': self._help,
            'train': self._train_test_infer,
            'test': self._train_test_infer,
            'infer': self._train_test_infer,
            'new-project': self._new_project,
        }
        self.mode = sys.argv[1] if len(sys.argv) > 1 else 'help'

    def _help(self):
        # TODO: this is a temporary help. find what's the best convention and way to do it
        msg = "\nGANSLATE COMMAND LIST\n\n"
        msg += "help            List all available commands and info on them.\n"
        msg += "train           Run training.\n"
        msg += "test            Test a trained model. Possible only with paired dataset.\n"
        msg += "infer           Perform inference with a trained model.\n"
        msg += "new-project     Generate a project template. Specify the name and the path where it should be generated\n"
        sys.stdout.write(msg + "\n")

    def _train_test_infer(self):
        engine = init_engine(self.mode)
        engine.run()

    def _new_project(self):
        raise NotImplementedError

    def run(self):
        if self.mode not in self.COMMANDS.keys():
            sys.stderr.write(f"\nUnknown command `{self.mode}`\n")
            self.mode = 'help'

        command = self.COMMANDS[self.mode]
        command()
