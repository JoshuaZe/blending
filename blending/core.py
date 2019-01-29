import cv2 as cv
import numpy as np


def rotate_simple(image, rotation_angle):
    """image rotation.

        rotating one single image.

        Args:
            image: An RGBA rectangle image.
            rotation_angle: the angle rotated of figure image.
                the positive value means anti-clockwise rotation.

        Returns:
            A rotated image.
    """
    (h, w) = image.shape[:2]
    rotate_center = (w // 2, h // 2)
    # perform the rotation Matrix
    M = cv.getRotationMatrix2D(rotate_center, rotation_angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated


def rotate(image, rotation_angle):
    """image rotation with boundary changing.

        rotating one single image.

        Args:
            image: An RGBA rectangle image.
            rotation_angle: the angle rotated of figure image.
                the positive value means anti-clockwise rotation.

        Returns:
            A rotated image.
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv.warpAffine(image, M, (nW, nH))
    return rotated


def gen_alpha_mask_figure(base_shape, alpha_figure, offset_position):
    """generation alpha mask for blending.

        the alpha mask (or called transparency mask) should be the same shape of background image
        (except the channel), and also contains the offset and cropping of figure.

        Args:
            base_shape: An background image shape (height, width).
            alpha_figure: An RGBA alpha mask (or called transparency mask) rectangle image after rotation,
                the alpha (or A) is used as mask (fully transparent part is the covered part),
                if A is fully opaque, the whole RGB image will be showed.
            offset_position: the coordinate position of center of figure image (x, y)
                by using the cartesian coordinate system of base image.
                the upper left corner of base image is the (0,0) point,
                the abscissa (horizontal axis) is the first value designating the x-direction
                and the ordinate (vertical axis) is the second value designating the y-direction.

        Returns:
            A tuple include mask figure and figure position in mask:
            -   A alpha mask for figure that compared with background image.
            -   (top, bottom, left, right)

        Raises:

    """
    (base_h, base_w) = base_shape
    (figure_h, figure_w) = alpha_figure.shape[:2]
    (cX, cY) = offset_position
    # cropping and bordering
    top = cY - figure_h // 2
    bottom = base_h - (cY + figure_h // 2)
    left = cX - figure_w // 2
    right = base_w - (cX + figure_w // 2)
    # for bordering
    border_top = max(top, 0)
    border_bottom = max(bottom, 0)
    border_left = max(left, 0)
    border_right = max(right, 0)
    alpha_mask = cv.copyMakeBorder(alpha_figure, border_top, border_bottom, border_left, border_right,
                                   cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
    # for cropping
    crop_top = max(-top, 0)
    crop_bottom = crop_top + base_h
    crop_left = max(-left, 0)
    crop_right = crop_left + base_w
    alpha_mask = alpha_mask[crop_top:crop_bottom, crop_left:crop_right]
    # update position
    figure_position = (border_top, base_h - border_bottom, border_left, base_w - border_right)
    return alpha_mask, figure_position


def alpha_blending(base_image, alpha_mask):
    """image blending by alpha function.

        blending one single image into background image.
        https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/

        Args:
            base_image: An RGB background rectangle image.
            alpha_mask: An RGBA alpha mask (or called transparency mask) rectangle image,
                the alpha (or A) is used as mask (fully transparent part is the covered part),
                if A is fully opaque, the whole RGB image will be showed.

        Returns:
            A blended image.

    """
    mask = alpha_mask.copy()
    alpha = mask[:, :, :3]
    alpha[:, :, 0] = alpha_mask[:, :, 3]
    alpha[:, :, 1] = alpha_mask[:, :, 3]
    alpha[:, :, 2] = alpha_mask[:, :, 3]

    foreground = alpha_mask[:, :, :3]
    foreground = foreground.astype(float)
    background = base_image.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    blended = cv.add(foreground, background)

    return blended.astype(np.uint8)


def poisson_blending_opencv(base_image, alpha_mask, offset_position):
    """image blending by poisson function in opencv.

        blending one single image into background image.
        https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/

        Args:
            base_image: An RGB background rectangle image.
            alpha_mask: An RGBA alpha mask (or called transparency mask) rectangle image,
                the alpha (or A) is used as mask (fully transparent part is the covered part),
                if A is fully opaque, the whole RGB image will be showed.
            offset_position: the coordinate position of center of figure image (x, y)
                by using the cartesian coordinate system of base image.
                the upper left corner of base image is the (0,0) point,
                the abscissa (horizontal axis) is the first value designating the x-direction
                and the ordinate (vertical axis) is the second value designating the y-direction.

        Returns:
            A blended image.

    """
    mask = alpha_mask[:, :, 3]
    foreground = alpha_mask[:, :, :3]
    blended = cv.seamlessClone(src=foreground, dst=base_image, mask=mask, p=offset_position, flags=cv.NORMAL_CLONE)

    return blended


def poisson_RGB_blending(target, source, mask, result, figure_position, region=1):
    """core image blending by poisson RGB blending function (write by ourselves) .

        blending one single image into background image.

        Args:
            target: An RGB background rectangle image.
            source: An RGB blending rectangle image (same shape as target).
            mask: An alpha mask (only alpha channel, same shape as source).
            result: initialization result by alpha blending.
            figure_position: (top, bottom, left, right) tuple of the figure position.
                used for reducing looping of alpha mask source image.
            region: only an epsilon region around mask can be adjusted.

        Returns:
            A blended image.

    """
    (top, bottom, left, right) = figure_position
    (base_h, base_w) = target.shape[:2]
    previous_epsilon = 1.0
    cnt = 0
    while True:
        dx = 0
        absx = 0
        for y in range(top, bottom):
            for x in range(left, right):
                # only an epsilon region around mask can be adjusted
                flag_mask = False
                if mask[y, x] > 0:
                    region_points = [
                        (min(x + region, base_w - 1), y), (max(x - region, 0), y),
                        (x, min(y + region, base_h - 1)), (x, max(y - region, 0))
                    ]
                    flag_mask = any([mask[e_y, e_x] == 0 for (e_x, e_y) in region_points])
                if flag_mask:
                    # inside omega
                    neigbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    num_neighbors = len(neigbors)
                    # for each channel
                    for rgb in range(0, 3):
                        sum_fq = 0
                        sum_vpq = 0
                        sum_boundary = 0
                        for (k_x, k_y) in neigbors:
                            if k_y < 0 or k_y > base_h - 1 or k_x < 0 or k_x > base_w - 1:
                                # neigbors does not exist
                                continue
                            flag_omega = mask[k_y, k_x] > 0
                            if flag_omega:
                                # inside omega
                                sum_fq = sum_fq + result[k_y, k_x, rgb]
                            else:
                                # outside omega
                                sum_boundary = sum_boundary + target[k_y, k_x, rgb]
                            replace_target = int(target[y, x, rgb]) - int(target[k_y, k_x, rgb])
                            replace_source = int(source[y, x, rgb]) - int(source[k_y, k_x, rgb])
                            if not flag_omega & abs(replace_target) >= abs(replace_source):
                                sum_vpq = sum_vpq + replace_target
                            else:
                                sum_vpq = sum_vpq + replace_source
                        # average all neigbors with gradient consider
                        new_value = (sum_fq + sum_boundary + sum_vpq) / num_neighbors
                        dx = dx + abs(new_value - result[y, x, rgb])
                        absx = absx + abs(new_value)
                        result[y, x, rgb] = new_value
        cnt = cnt + 1
        epsilon = dx / max(absx, 1e-5)
        print(cnt, epsilon, sep=":")
        # convergence
        if epsilon <= 1e-1 or previous_epsilon - epsilon <= 1e-2:
            break
        else:
            previous_epsilon = epsilon
    return result


def poisson_blending(base_image, alpha_mask, figure_position):
    """image blending by poisson function (write by ourselves) .

        blending one single image into background image.

        Args:
            base_image: An RGB background rectangle image.
            alpha_mask: An RGBA alpha mask (or called transparency mask) rectangle image,
                the alpha (or A) is used as mask (fully transparent part is the covered part),
                if A is fully opaque, the whole RGB image will be showed.
            figure_position: (top, bottom, left, right) tuple of the figure position.
                used for reducing looping of alpha mask source image.

        Returns:
            A blended image.

    """
    mask = alpha_mask[:, :, 3]
    source = alpha_mask[:, :, :3]
    # alpha blending
    init_blended = alpha_blending(base_image, alpha_mask)
    blended = poisson_RGB_blending(base_image, source, mask, init_blended, figure_position)
    return blended


def blending_images(base_image, alpha_figure, offset_position, rotation_angle, model_type="alpha_blending"):
    """image blending core function.

        blending one single rotated figure image into background image.

        Args:
            base_image: An RGB background rectangle image.
            alpha_figure: An RGBA alpha mask (or called transparency mask) rectangle image,
                the alpha (or A) is used as mask (fully transparent part is the covered part),
                if A is fully opaque, the whole RGB image will be showed.
            offset_position: the coordinate position of center of figure image (x, y)
                by using the cartesian coordinate system of base image.
                the upper left corner of base image is the (0,0) point,
                the abscissa (horizontal axis) is the first value designating the x-direction
                and the ordinate (vertical axis) is the second value designating the y-direction.
            rotation_angle: the angle rotated of figure image.
                the positive value means anti-clockwise rotation.
            model_type: two type blending model 'alpha_blending' (faster) and 'poisson_blending' (better).

        Returns:
            A blended image.

        Raises:

    """
    base_shape = base_image.shape[:2]
    rotation_figure = rotate(alpha_figure, rotation_angle=rotation_angle)
    alpha_mask, figure_position = gen_alpha_mask_figure(base_shape=base_shape, alpha_figure=rotation_figure,
                                                        offset_position=offset_position)
    if model_type == "poisson_blending":
        blended = poisson_blending(base_image, alpha_mask, figure_position)
    else:
        blended = alpha_blending(base_image, alpha_mask)
    return blended




if __name__ == '__main__':
    print("read background RGB image")
    base_image = cv.imread('../image_case/base_image.jpg', cv.IMREAD_COLOR)
    print(base_image.shape)
    print("read figure RGBA image")
    alpha_figure = cv.imread('../image_case/figure_image.jpg', cv.IMREAD_UNCHANGED)
    print(alpha_figure.shape)
    print("rotate figure RGBA image")
    rotation_figure = rotate(alpha_figure, rotation_angle=45)
    print(rotation_figure.shape)
    # transparency has no use in computer-vision, so imshow() just drops the alpha channel
    cv.imshow('image', rotation_figure)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("alpha mask generation")
    alpha_mask, figure_position = gen_alpha_mask_figure(base_shape=base_image.shape[:2],
                                                        alpha_figure=rotation_figure,
                                                        offset_position=(800, 900))
    print(alpha_mask.shape)
    cv.imshow('image', alpha_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("alpha blending")
    blended_by_alpha = alpha_blending(base_image, alpha_mask)
    # cv.imwrite('/Users/zezzhang/Workspace/poisson_blending/image_case/quick_out.jpg', blended_by_alpha)
    cv.imshow('image', blended_by_alpha)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # print("poisson blending by opencv")
    # blended_by_poisson = poisson_blending_opencv(base_image, alpha_mask, offset_position=(400, 500))
    # cv.imshow('image', blended_by_poisson)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print("poisson blending by us")
    blended_by_poisson = poisson_blending(base_image, alpha_mask, figure_position)
    # cv.imwrite('/Users/zezzhang/Workspace/poisson_blending/image_case/better_out.jpg', blended_by_poisson)
    cv.imshow('image', blended_by_poisson)
    cv.waitKey(0)
    cv.destroyAllWindows()
    blended = blending_images(base_image, alpha_figure, offset_position=(400, 500), rotation_angle=90, model_type="alpha_blending")
    cv.imshow('image', blended)
    cv.waitKey(0)
    cv.destroyAllWindows()

