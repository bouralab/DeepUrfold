import pprint
import subprocess
import torch
import MinkowskiEngine as ME
from pytorch_lightning.overrides.data_parallel import LightningDataParallel, LightningDistributedDataParallel

def scatter(inputs, kwargs, device_ids):
    assert len(inputs) == 2 and len(inputs[0]) == 3 and len(inputs[0][0]) == len(inputs[0][1]), "Inputs must contain features and labels (2 [{}]) and have the same number of minibatches ({}, {})".format(len(inputs[0]), len(inputs[0][0]), len(inputs[0][1]))
    assert len(inputs[0][0]) == len(device_ids), "Error with batch, there must be the same number of minibatches ({}) as there are GPUs ({})".format(len(inputs[0][0]), len(device_ids))

    #DataLoader splits them up, so only need to map to ids
    #return [((feat.to(device), labels), inputs[1]) for feat, labels, device in zip(inputs[0][0], inputs[0][1], device_ids)], None
    print(pprint.pformat(inputs[0][1]), "---")
    return [((coords, feat.to(device), labels), inputs[1]) for coords, feat, labels, device in zip(inputs[0][0], inputs[0][1], inputs[0][2], device_ids)], [{}]

def gather(outputs):
    device = next(iter(outputs[0].values())).device

    results = {}
    for key in outputs[0].keys():
        results[key] = torch.cat([x[key].to(device) for x in outputs])

    return results

class MinkowskiDataParallel(LightningDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter(inputs, kwargs, device_ids)
        # assert len(inputs) == 2 and len(inputs[0]) == 3 and len(inputs[0][0]) == len(inputs[0][1]), "Inputs must contain features and labels (2 [{}]) and have the same number of minibatches ({}, {})".format(len(inputs[0]), len(inputs[0][0]), len(inputs[0][1]))
        # assert len(inputs[0][0]) == len(device_ids), "Error with batch, there must be the same number of minibatches ({}) as there are GPUs ({})".format(len(inputs[0][0]), len(device_ids))
        #
        # #DataLoader splits them up, so only need to map to ids
        # #return [((feat.to(device), labels), inputs[1]) for feat, labels, device in zip(inputs[0][0], inputs[0][1], device_ids)], None
        # return [((coords, feat.to(device), labels), inputs[1]) for coords, feat, labels, device in zip(inputs[0][0], inputs[0][1], inputs[0][2], device_ids)], [{}]

    def gather(self, outputs):
        return gather(outputs)
        # device = next(iter(outputs[0].values())).device
        #
        # results = {}
        # for key in outputs[0].keys():
        #     results[key] = torch.cat([x[key].to(device) for x in outputs])
        #
        # return results

class MinkowskiDistributedDataParallel(LightningDistributedDataParallel):
    pass
    # def scatter(self, inputs, kwargs, device_ids):
    #     scattered_features, _ = super().scatter([inputs[0][1]], {}, device_ids)
    #     outputs = [[(inputs[0][0], scattered_features[0][0], inputs[0][2]), inputs[1]]]
    #     return outputs, [{}]
    #
    # def gather(self, outputs):
    #     print("GATHER IN: {}".format(pprint.pformat(outputs)))
    #     out = super().gather(outputs)
    #     print("GATHER OUT: {}".format(out))
    #     return out
        # device = next(iter(outputs[0].values())).device
        #
        # results = {}
        # for key in outputs[0].keys():
        #     results[key] = torch.cat([x[key].to(device) for x in outputs])
        #
        # return results
