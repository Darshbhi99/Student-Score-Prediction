from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements() -> List[str]:
    with open('requirements.txt', 'r') as f:
        res = [line.strip() for line in f.readlines()]
        if HYPEN_E_DOT in res:
            res.remove(HYPEN_E_DOT)
        return res


setup(
    name='score_prediction',
    version='0.0.1',
    author = "Darshan",
    author_email = "darshanbhiwapurkar@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
    )


