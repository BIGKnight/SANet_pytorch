import matplotlib.pyplot as plt


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()
