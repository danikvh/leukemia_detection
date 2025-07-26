import torch
import numpy as np
from typing import Tuple
from skimage.color import rgb2hed
from skimage import exposure
import matplotlib.pyplot as plt # For debug

class StainTransform:
    """
    Applies color deconvolution to an RGB image, separating Hematoxylin, Eosin, and DAB stains.
    Reconstructs a new 3-channel image suitable for Cellpose, typically with:
    - Green channel: Hematoxylin (nuclei)
    - Blue channel: Combined Eosin and DAB (cytoplasm/whole cell)

    Parameters:
        gamma_value (float): Gamma correction for contrast enhancement.
        dab (float): Weight for the DAB stain component in the blue channel.
        eosin (float): Weight for the Eosin stain component in the blue channel.
        only_nuclei (bool): If True, the output image will only contain the nuclei (Green channel),
                            with Red and Blue channels set to zero.
        inversion (bool): If True, inverts the stain intensities (useful if stains appear dark
                          and bright objects are desired for Cellpose).
        debug (bool): If True, displays intermediate and final debug images using matplotlib.
    """
    def __call__(self, img_tensor: torch.Tensor, mask: torch.Tensor,
                 gamma_value: float = 1, dab: float = 1.0, eosin: float = 0.0,
                 only_nuclei: bool = False, inversion: bool = False, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        image = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        if image.dtype != np.uint8:
            if image.max() > 1.0:
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)

        image_float = image.astype(np.float64) / 255.0
        hed = rgb2hed(image_float)

        h_stain_amount = hed[:, :, 0]
        h_normalized = exposure.rescale_intensity(h_stain_amount, out_range=(0.0, 1.0))
        if inversion:
            h_normalized = 1.0 - h_normalized
        h_gamma = exposure.adjust_gamma(h_normalized, gamma=gamma_value)
        hematoxylin_uint8 = (h_gamma * 255).astype(np.uint8)

        e_stain_amount = hed[:, :, 1]
        e_normalized = exposure.rescale_intensity(e_stain_amount, out_range=(0.0, 1.0))
        if inversion:
            e_normalized = 1 - e_normalized
        e_gamma = exposure.adjust_gamma(e_normalized, gamma=gamma_value)
        eosin_uint8 = (e_gamma * 255).astype(np.uint8)

        dab_stain_amount = hed[:, :, 2]
        dab_normalized = exposure.rescale_intensity(dab_stain_amount, out_range=(0.0, 1.0))
        if inversion:
            dab_normalized = 1 - dab_normalized
        dab_gamma = exposure.adjust_gamma(dab_normalized, gamma=gamma_value)
        dab_uint8 = (dab_gamma * 255).astype(np.uint8)

        combined_blue = eosin * e_normalized + dab * dab_normalized
        combined_blue = np.clip(combined_blue, 0.0, 1.0)
        combined_blue = exposure.adjust_gamma(combined_blue, gamma=gamma_value)
        combined_blue_uint8 = (combined_blue * 255).astype(np.uint8)

        nuclei_tensor = torch.from_numpy(hematoxylin_uint8).to(dtype=img_tensor.dtype, device=img_tensor.device)
        cyto_tensor = torch.from_numpy(combined_blue_uint8).to(dtype=img_tensor.dtype, device=img_tensor.device)

        H, W = img_tensor.shape[1:]
        rgb = torch.zeros((3, H, W), dtype=img_tensor.dtype, device=img_tensor.device)
        
        rgb[1] = nuclei_tensor
        
        if not only_nuclei:
            rgb[2] = cyto_tensor

        if debug:
            print("\n--- StainTransform Debug Output ---")
            def _display_debug_image(img_t, title):
                img_np_display = img_t.permute(1, 2, 0).cpu().numpy() if img_t.ndim == 3 else img_t.cpu().numpy()
                if img_np_display.max() > 1.01:
                    img_np_display = img_np_display.astype(np.uint8)
                else:
                    img_np_display = (img_np_display * 255).astype(np.uint8)
                plt.figure(figsize=(4, 4))
                plt.imshow(img_np_display)
                plt.title(title)
                plt.axis("off")
                plt.show()

            _display_debug_image(rgb, "Final RGB (StainTransform)")
            print(f"Hematoxylin: min={hematoxylin_uint8.min()}, max={hematoxylin_uint8.max()}")
            _display_debug_image(torch.from_numpy(hematoxylin_uint8).unsqueeze(0), "Hematoxylin Channel")
            print(f"Eosin: min={eosin_uint8.min()}, max={eosin_uint8.max()}")
            _display_debug_image(torch.from_numpy(eosin_uint8).unsqueeze(0), "Eosin Channel")
            print(f"DAB: min={dab_uint8.min()}, max={dab_uint8.max()}")
            _display_debug_image(torch.from_numpy(dab_uint8).unsqueeze(0), "DAB Channel")

            weights = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
            for w1, w2 in weights:
                combined_blue_debug = w1 * e_normalized + w2 * dab_normalized
                combined_blue_debug = np.clip(combined_blue_debug, 0.0, 1.0)
                combined_blue_debug = exposure.adjust_gamma(combined_blue_debug, gamma=gamma_value)
                combined_blue_uint8_debug = (combined_blue_debug * 255).astype(np.uint8)
                
                rgb_test = torch.zeros_like(rgb)
                rgb_test[1] = nuclei_tensor
                rgb_test[2] = torch.from_numpy(combined_blue_uint8_debug).to(dtype=img_tensor.dtype, device=img_tensor.device)
                print(f"COMBINED {w1} EOSIN AND {w2} DAB FOR BLUES RGB (Debug Test)")
                _display_debug_image(rgb_test, f"COMBINED {w1} EOSIN AND {w2} DAB")
            print("--- End StainTransform Debug Output ---\n")

        return rgb, mask