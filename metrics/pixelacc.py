import numpy as np


def pixel_accuracy(eval_segm, gt_segm):
    """Calculates the pixel accuracy

    Args:
        eval_segm (np.array): The segmented mask from the generated image
        gt_segm (np.array): The ground truth segmentation mask

    Returns:
        float: pixel accuracy
    """
    assert eval_segm.shape == gt_segm.shape, "Segmentation mask dimensions don't match"
    comparison = (eval_segm == gt_segm).astype(int)
    res = np.prod(comparison,axis=2)
    return np.mean(res)


#Test
test_segm = np.random.randint(2,size=(300,300,3))
test_gt = np.random.randint(2,size=(300,300,3))

acc = pixel_accuracy(test_segm,test_gt)
print("Accuracy",acc)
