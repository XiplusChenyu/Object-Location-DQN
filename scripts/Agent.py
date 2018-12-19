from Setting import *

'''This part describe the movement of the agent'''


def agent_move_mask(action, original_shape, mask_size, offset, region_image):
    region_mask = np.zeros(original_shape)
    mask_size = (mask_size[0] * scale_subregion, mask_size[1] * scale_subregion)

    offset_aux = (0, 0)
    if action == 2:
        offset_aux = (0, mask_size[1] * scale_mask)
        offset = (offset[0], offset[1] + mask_size[1] * scale_mask)
    elif action == 3:
        offset_aux = (mask_size[0] * scale_mask, 0)
        offset = (offset[0] + mask_size[0] * scale_mask, offset[1])
    elif action == 4:
        offset_aux = (mask_size[0] * scale_mask,
                      mask_size[1] * scale_mask)
        offset = (offset[0] + mask_size[0] * scale_mask,
                  offset[1] + mask_size[1] * scale_mask)
    elif action == 5:
        offset_aux = (mask_size[0] * scale_mask / 2,
                      mask_size[1] * scale_mask / 2)
        offset = (offset[0] + mask_size[0] * scale_mask / 2,
                  offset[1] + mask_size[1] * scale_mask / 2)

    # @Update the status here, crop  the image, and update the region mask:
    region_image = region_image[int(offset_aux[0]):int(offset_aux[0] + mask_size[0]),
                                int(offset_aux[1]): int(offset_aux[1] + mask_size[1])]  # Crop the image

    region_mask[int(offset[0]): int(offset[0] + mask_size[0]),
                int(offset[1]): int(offset[1] + mask_size[1])] = 1  # Mask the area we chopped

    return region_image, region_mask, offset, mask_size










