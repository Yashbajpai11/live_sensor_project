from setuptools import setup, find_packages


def get_requirements()->list[str]:
    requirement_list : list[str] = []

    return requirement_list


setup(
    name = 'sensor',
    version='0.0.1',
    author='yash',
    author_email= 'yashbajpai2003@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements(),
)