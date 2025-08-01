def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
    elif name == "aper_vpt":
        from models.aper_vpt import Learner 
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == 'slca':
        from models.slca import Learner
    elif name == 'lae':
        from models.lae import Learner
    elif name == 'fecam':
        from models.fecam import Learner
    elif name == 'dgr':
        from models.dgr import Learner
    elif name == 'slca_pp':
        from models.slca_pp import Learner
    elif name == 'mos':
        from models.mos import Learner
    elif name == 'cofima':
        from models.cofima import Learner
    elif name == 'ssiat':
        from models.ssiat import Learner
    else:
        assert 0
    return Learner(args)