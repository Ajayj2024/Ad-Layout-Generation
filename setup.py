from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'
def install_requirements(requirement_file: str) -> list[str]:
    '''
    Returns the list of libraries mentioned in requirements.txt
    '''
    requirements = []
    with open(requirement_file) as f:
        libraries = f.readlines()
        requirements = [l.replace('\n','') for l in libraries]
    
    # Remove -e . present in requirements.txt 
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name= "Ad Creative Layout Generation",
    version= "0.0.0",
    author= "Ajay",
    packages= find_packages(), # find_packages() checks the the folder which has __init__.py folder which is considered as package
    install_requires= install_requirements('requirements.txt')
)