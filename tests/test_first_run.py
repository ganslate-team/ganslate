from pathlib import Path

from ganslate.utils.cli.interface import setup_first_run
from ganslate.engines.utils import init_engine

# test_with_pytest.py
def test_first_run():
    """[summary]
    Generate setup for first run and check if the directory and files are
    created.
    """
    setup_first_run(".", True, extra_context={"number_of_iterations": 2, 
                                              "project_name": "first_run_test",
                                              "logging_frequency": 1,
                                              "enable_cuda": False
                                             })

    generated_project_dir = Path("first_run_test")
    assert generated_project_dir.is_dir()
    assert (generated_project_dir / "facades" / "train" / "A" ).is_dir()
    assert (generated_project_dir / "facades" / "train" / "B" ).is_dir()


def test_training():
    """[summary]
    Run 10 iterations of dummy training and see if it works.
    """
    assert init_engine('train', ["config=first_run_test/default.yaml"]).run()
