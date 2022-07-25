import torch

import MinkowskiEngine as ME
if ME.__version__ == "0.4.3":
    from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiModuleBase as MinkowskiNonlinearityBase
else:
    from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiNonlinearityBase

class MinkowskiLeakyReLU(MinkowskiNonlinearityBase):
    MODULE = torch.nn.LeakyReLU
