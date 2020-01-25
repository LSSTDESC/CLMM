
def install_clmm_pipeline(upgrade=False):
    """
    Tries to import clmm module. If this fails, it installs it and cluster_toolkit
    """
    import sys
    try:
        import clmm
        import cluster_toolkit
        installed = True
    except ImportError:
        installed = False
        if not upgrade:
             print('clmm is already installed and upgrade is False')
        else:
            !{sys.executable} -m pip install --user --upgrade git+https://github.com/tmcclintock/cluster_toolkit.git
            !{sys.executable} -m pip install --user --upgrade git+https://github.com/LSSTDESC/CLMM