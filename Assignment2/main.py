
import numpy as np
from Utils.data_utils import load_images
from Utils.layer_utils import Test_conv_forward, Test_conv_forward_IP, Test_conv_backward
from Utils.layer_utils import Test_max_pool_forward, Test_max_pool_backward

Sample_images = load_images()


def main():
    # Compare your convolution forward outputs to ours; difference should be around 4e-8
    print('Testing conv_forward for padding=valid')
    print('difference: %e\n' % Test_conv_forward(1))

    print('Testing conv_forward for padding=same')
    print('difference: %e\n' % Test_conv_forward(2))

    ## Compare your max pooling forward outputs to ours; difference should be around 6e-8
    print('Testing max_pool_forward')
    print('difference: %e' % Test_max_pool_forward())

    # Compare your convolution backward outputs to ours; difference should be around 1e-9
    print('Testing conv_backward for padding=valid')
    Diff1 = Test_conv_backward(1)
    print('difference dx: %e' % Diff1[0])
    print('difference dw: %e' % Diff1[1])
    print('difference db: %e\n' % Diff1[2])

    print('Testing conv_backward for padding=same')
    Diff2 = Test_conv_backward(2)
    print('difference dx: %e' % Diff2[0])
    print('difference dw: %e' % Diff2[1])
    print('difference db: %e\n' % Diff2[2])

    # Compare your max pooling backward outputs to ours; difference should be around 1e-12
    print('Testing max_pool_backward')
    print('difference dx: %e' % Test_max_pool_backward())


if __name__ == '__main__':
    main()
