import model.my

from torchinfo import summary


net = model.my.Unet3x3(114, 18)
summary(net, input_size=(1, 114, 3, 3))

net = model.my.Wisppn(114, 18)
summary(net, input_size=(1, 150, 3, 3))

net = model.my.Piw(150, 18, 38)
summary(net, input_size=(1, 150, 3, 3))

net = model.my.Wpnet(114, 18)
summary(net, input_size=(1, 30, 3, 3))
