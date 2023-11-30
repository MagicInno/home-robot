# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

# 用于对二值图像进行膨胀操作。增加图像中白色（或高值）区域的面积，通常用于连接相邻对象、填充小的空洞和平滑边缘。
def binary_dilation(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        # 二值图像张量，其形状为 (bs, 1, H1, W1)，其中  bs 表示批次大小，H1 和 W1 分别是图像的高度和宽度。
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)
        # 二值结构元素张量，其形状为 (1, 1, H2, W2)，用于定义膨胀操作的邻域和形状。H2 和 W2 是结构元素的高度和宽度。

    Returns:
        binary image tensor of the same shape as input
膨胀操作：函数使用 torch.nn.functional.conv2d 来执行膨胀操作。这里的卷积实际上是将结构元素 kernel 应用于 binary_image，以确定哪些区域应该被膨胀。
边界填充：通过设置 padding=kernel.shape[-1] // 2，确保输出图像与输入图像有相同的尺寸。这种填充方式使得结构元素的中心与每个像素对齐。
限制输出范围：使用 torch.clamp 函数将输出限制在0和1之间，以保证结果仍然是二值的。
    """
    return torch.clamp(
        torch.nn.functional.conv2d(binary_image, kernel, padding=kernel.shape[-1] // 2),
        0,
        1,
        
    )
    # 函数返回膨胀后的二值图像张量，其形状与输入图像相同。

# 对二值图像进行侵蚀操作。
def binary_erosion(binary_image, kernel):
    """
    侵蚀是图像处理中的另一种形态学操作，用于减少图像中白色（或高值）区域的面积。这通常用于去除小的白色噪点、断开相邻对象或缩小对象的边界。
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
        
    1 - binary_image 反转图像中的白色和黑色区域
    """
    return 1 - torch.clamp(
        torch.nn.functional.conv2d(
            1 - binary_image, kernel, padding=kernel.shape[-1] // 2
        ),
        0,
        1,
    )

# 二值开运算 (binary_opening):首先对图像执行侵蚀操作，然后对结果执行膨胀操作。
# 用于去除小对象或噪点，并断开对象之间的狭窄连接，同时保持较大对象的大小基本不变。
def binary_opening(binary_image, kernel):
    return binary_dilation(binary_erosion(binary_image, kernel), kernel)

# 二值闭运算 (binary_closing):首先对图像执行膨胀操作，然后对结果执行侵蚀操作。
# 用于填充对象内的小空洞和缝隙，以及连接靠近的对象，但不显著改变对象的总体大小。
def binary_closing(binary_image, kernel):
    return binary_erosion(binary_dilation(binary_image, kernel), kernel)

# 二值去噪 (binary_denoising):结合了闭运算和开运算。首先执行闭运算以填充空洞和连接近邻对象，然后执行开运算以去除由闭运算可能引入的小噪点和细节。
# 特别适用于去除噪声和清理图像，同时保持主要对象的结构。
def binary_denoising(binary_image, kernel):
    return binary_opening(binary_closing(binary_image, kernel), kernel)

# 从给定的 PyTorch 张量中提取边缘
# 使用索贝尔（Sobel）算子来检测图像中的边缘，并根据指定的阈值返回一个二值图像，其中边缘区域被标记为 True。
def get_edges(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Extract edges from a torch tensor."""

    mask = mask.float()
    # 将输入的 mask 张量转换为浮点类型以进行后续的数学运算。

    # Define the Sobel filter kernels
    # 创建两个索贝尔核（sobel_x 和 sobel_y），分别用于检测水平方向（x方向）和垂直方向（y方向）的边缘。
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device
    )
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device
    )

    # Calculate padding for convolution to preserve the original size
    # 计算在卷积过程中所需的填充，以保持图像尺寸不变。
    # Sobel x and y operators are the same size
    padding_x = sobel_x.size(0) // 2
    padding_y = sobel_x.size(1) // 2

    # Apply Sobel filter to detect edges in x and y directions
    # 使用索贝尔滤波器对图像进行卷积，分别在 x 方向和 y 方向上检测边缘。
    edges_x = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        sobel_x.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )
    edges_y = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        sobel_y.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )

    # Combine x and y edge responses to get the magnitude of edges
    # 通过结合 x 方向和 y 方向的边缘响应来计算边缘的幅度
    # 应用阈值 threshold 来确定最终的边缘。边缘幅度高于阈值的区域被视为边缘。
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    edges = edges[0, 0] > threshold
    assert (
        edges.shape == mask.shape
    ), "something went wrong when computing padding, most likely - shape not preserved"
    return edges
    # 返回一个与输入 mask 形状相同的二值图像张量，其中包含了检测到的边缘。

# 在 PyTorch 中扩展（膨胀）一个二值掩码。
# 将掩码中的白色（或高值）区域向外扩展指定的半径。
def expand_mask(mask: torch.Tensor, radius: int, threshold: float = 0.5):
    """Expand a mask by some radius in pytorch"""

    # Needs to be converted to a float to work
    mask = mask.float()
    
    # Create a disk-shaped structuring element
    # 使用 torch.meshgrid 生成一个二维网格，然后创建一个半径为 radius 的圆盘形状的二值结构元素 selem
    x, y = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    selem = (x**2 + y**2 <= radius**2).to(torch.float32)

    # Calculate padding for convolution to preserve the original size
    # 计算在卷积过程中所需的填充大小，以保持图像尺寸不变。
    padding_x = selem.size(0) // 2
    padding_y = selem.size(1) // 2

    # Apply binary dilation to expand the mask
    # 使用 F.conv2d 对掩码进行卷积，应用圆盘形状的结构元素进行膨胀操作。这样可以扩大掩码中白色区域的面积。
    expanded_mask = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        selem.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )

    # Binarize the expanded mask (optional)
    # 将膨胀后的掩码二值化，以确保结果仍然是二值的。这里还使用了阈值 threshold 来判断哪些区域应该被视为掩码的一部分。
    expanded_mask = (expanded_mask > 0).to(torch.float32)
    expanded_mask = expanded_mask[0, 0] > threshold
    assert (
        expanded_mask.shape == mask.shape
    ), "something went wrong when computing padding, most likely - shape not preserved"
    return expanded_mask

# 在一个二值掩码中找到离指定点最近的点。
def find_closest_point_on_mask(mask: torch.Tensor, point: torch.Tensor):
    """
    Find the closest point on a binary mask to another point.

    Parameters:
    - mask: Binary mask where 1 represents the region of interest (PyTorch tensor).
    - point: Coordinates of the target point (PyTorch tensor).

    Returns:
    - closest_point: Coordinates of the closest point on the mask (PyTorch tensor).
    """
    # Ensure the input mask is binary (0 or 1)
    # 确保掩码 mask 是二值的，即只包含 0 和 1。
    mask = (mask > 0).to(torch.float32)
    

    # Find all nonzero (1) pixels in the mask
    # 找到掩码中的非零像素：
    nonzero_pixels = torch.nonzero(mask, as_tuple=False)

    if nonzero_pixels.size(0) == 0:
        # If the mask has no nonzero pixels, return None
        return None

    # Calculate the Euclidean distance between the target point and all nonzero pixels
    # 处理掩码全零的情况：
    distances = torch.norm(nonzero_pixels - point, dim=1)

    # Find the index of the closest pixel
    # 计算 point 和掩码中每个非零像素之间的欧几里得距离。
    closest_index = torch.argmin(distances)

    # Get the closest point
    # 找到距离最小的像素的索引，即离 point 最近的点。
    closest_point = nonzero_pixels[closest_index]

    return closest_point
