import torch

def NC(water_ori, water_de):

    quarm_a = torch.sum((water_ori-torch.mean(water_ori)) * (water_de-torch.mean(water_de)))
    # quarm_b1 = torch.std(water_ori)
    # quarm_b2 = torch.std(water_de)
    quarm_b1 = torch.sqrt(torch.sum((water_ori-torch.mean(water_ori))**2))
    quarm_b2 = torch.sqrt(torch.sum((water_de-torch.mean(water_de))**2))
    nc = quarm_a/(quarm_b1*quarm_b2)
    return nc


if __name__ == '__main__':
    for i in range(100):
        a = torch.rand(32, 32)
        b = torch.rand(32, 32)
        print(NC(a, b))








