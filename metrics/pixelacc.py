import numpy as np


def pixel_accuracy(eval_segm, gt_segm):
    """Calculates the pixel accuracy

    Args:
        eval_segm (np.array): The segmented mask from the generated image
        gt_segm (np.array): The ground truth segmentation mask

    Returns:
        float: pixel accuracy
    """

    assert eval_segm.shape == gt_segm.shape
    size = eval_segm.shape
    num_pixels = np.prod(size)
    eval_segm.flatten()
    gt_segm.flatten()
    return np.sum(eval_segm == gt_segm)/num_pixels


#Test
test_segm = np.random.randint(1034,size=(300,300,3))
test_gt = np.random.randint(1034,size=(300,300,3))

acc = pixel_accuracy(test_segm,test_gt)
print("Accuracy",acc)