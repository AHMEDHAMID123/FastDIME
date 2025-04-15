from torchvision.transforms import transforms, ToPILImage
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import yaml
from functools import partial
import os
from typing import Union
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
from torch.nn import functional as F


class fastdime_config(BaseModel):

    base_dir: str
    sd_model: str
    guidance_scale: float
    num_inference_steps: int
    image_path: str
    seed: Union[int, None]
    dtype: str
    image_path: str
    prompt: str
    edit_prompt: str
    font_size: int
    strength: float


class FastDIME:
    def __init__(self, config: fastdime_config, device: torch.device):
        self.device = device
        self.dtype = torch.float16 if config.dtype == "half" else torch.float32
        self.config = config
        self.piprline = StableDiffusion3Pipeline.from_pretrained(
            self.config.sd_model,
            torch_dtype=self.dtype,
            text_encoder_3=None,
            tokenizer_3=None,
        ).to(device)

        self.vae_enocder = self.piprline.vae.encode
        self.vae_decoder = self.piprline.vae.decode
        self.scaling_factor = self.pipeline.vae.config.scaling_factor
        self.shift_factor = self.pipeline.vae.config.shift_factor

    @torch.no_grad()
    def latent_encoder(self, img):
        """
        Encode an image into latent space.

        Args:
            img (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Latent representation of the image.
        """
        if isinstance(img, Image.Image):
            transform = transforms.ToTensor()
            im_tensor = transform(img)
        else:
            im_tensor = img
        # preprocessing in sd3
        im_tensor = 2.0 * im_tensor - 1.0
        latent_image = self.vae_encoder(
            im_tensor.unsqueeze(0).to(self.dtype).to(self.device)
        )
        latent_model_input = latent_image.latent_dist.sample()
        latent_model_input = (
            latent_model_input - self.shift_factor
        ) * self.scaling_factor
        return latent_model_input

    @torch.no_grad()
    def latent_decoder(
        self,
        latent: torch.tensor,
        img: bool = False,
    ):
        """
        Decode a latent representation back into an image.

        Args:
            latent (torch.Tensor): Latent representation.
            img (bool): Whether to return a PIL image.

        Returns:
            torch.Tensor or PIL.Image.Image: Decoded image.
        """
        latent = (latent / self.scaling_factor) + self.shift_factor

        decoded = self.vae_decoder(latent)
        decoded = (decoded.sample / 2.0 + 0.5).clamp(0, 1)[0]
        if img:
            to_pil = ToPILImage()
            img = to_pil(decoded.squeeze(0))
            return img
        return decoded

    def prepare_time_steps(self, num_inference_steps=10, end_time=1000, shift=3):
        """
        Prepare time steps.

        Args:
            num_inference_steps (int): Number of inference steps.
            end_time (int): End time for the process.
            shift (int): Shift factor for time step scaling.

        Returns:
            torch.Tensor: Time steps.
        """
        num_training_steps = 1000
        timesteps = torch.linspace(1, end_time, num_training_steps).flip(0)
        sigmas = timesteps / num_training_steps

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()
        timesteps = torch.linspace(
            num_training_steps * sigma_max,
            num_training_steps * sigma_min,
            num_inference_steps,
        )
        sigmas = timesteps / num_training_steps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        return sigmas

    def rescale_time_step(self, timestep):
        """
        Rescale a time step to the stable diffusion range.

        Args:
            timestep (float): Time step to rescale.

        Returns:
            float: Rescaled time step.
        """
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1000.0
        scaled = (timestep - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
        return scaled

    def generate_image(self, prompt):
        """
        Generate an image from a text prompt.

        Args:
            prompt (str): Text prompt for image generation.

        Returns:
            PIL.Image.Image: Generated image.
        """
        return self.pipeline(prompt).images[0]

    @torch.enable_grad()
    def clean_multiclass_cond_fn(
        self,
        x_t,
        y,
        resize,
        classifier,
        s,
        use_logits,
        predictor_img_size,
        lr,
        momentum,
        optimizer,
        threshold: int = None,
    ):
        classifier.eval()
        x_in = nn.Parameter(x_t.detach(), requires_grad=True)
        x_in = [x_in]
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(
                x_in,
                lr=lr,
                momentum=momentum,
            )
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(x_in, lr=lr)
        optimizer.zero_grad()
        if resize:
            x_img = x_in[0]
        else:
            x_img = self.latent_decoder(x_in[0])
        # test_transform = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )

        # train_transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(224),  # Random crop and resize
        #         transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406],  # ImageNet normalization mean
        #             std=[0.229, 0.224, 0.225],
        #         ),  # ImageNet normalization std
        #     ]
        # )
        # x_img = test_transform(x_img)
        x_img = transforms.Resize(predictor_img_size)(x_img)

        classifier.eval()
        # classifier.train()
        selected = classifier(x_img)
        # classifier.eval()
        # # # Select the target logits
        if not use_logits:
            selected = F.log_softmax(selected, dim=1)
        selected = -selected[range(len(y)), y]
        selected = selected * s
        grads = torch.autograd.grad(selected.sum(), x_in)[0]
        # loss = torch.nn.CrossEntropyLoss()
        # logits_gradient = classifier(x_img) / s
        # print("target label", y)
        # losses = loss(logits_gradient, y.to("cuda"))
        # losses.backward()
        # optimizer.step()
        # grads = x_in[0].grad
        # grads = torch.autograd.grad(losses, x_in)[0]
        # breakpoint()
        # self.latent_decoder(x_in[0], img=True).save(
        #     "/home/a.zeid/peal_private/utils/delete_me/decoded_parameters.png"
        # )
        # self.latent_decoder(grads, img=True).save(
        #     "/home/a.zeid/peal_private/utils/delete_me/grads_parameters.png"
        # )
        if threshold:
            max_values = grads.abs().amax(dim=(2, 3), keepdim=True)
            # max_values = grads.abs().max()
            grads[grads < threshold * max_values] = 0.0
        return grads

    @torch.enable_grad()
    def clean_class_cond_fn(x_t, y, classifier, s, use_logits):
        """
        Computes the classifier gradients for the guidance

        :param x_t: clean instance
        :param y: target
        :param classifier: classification model
        :param s: scaling classifier gradients parameter
        :param use_logits: compute the loss over the logits
        """

        x_in = x_t.detach().requires_grad_(True)
        logits = classifier(x_in)

        y = y.to(logits.device).float()
        # Select the target logits,
        # for those of target 1, we take the logits as they are (sigmoid(logits) = p(y=1 | x))
        # for those of target 0, we take the negative of the logits (sigmoid(-logits) = p(y=0 | x))
        if len(logits.shape) == 2:
            selected = torch.nn.CrossEntropyLoss()(logits, y.long())

        else:
            selected = y * logits - (1 - y) * logits

            if use_logits:
                selected = -selected

            else:
                selected = -F.logsigmoid(selected)

        selected = selected * s
        grads = torch.autograd.grad(selected.sum(), x_in)[0]

        return grads

    @torch.enable_grad()
    def dist_cond_fn(
        self,
        x_tau,
        z_t,
        x_t,
        alpha_t,
        l1_loss,
        l2_loss,
        l_perc,
        scale_grads,
        lr,
        momentum,
    ):
        """
        Computes the distance loss between x_t, z_t and x_tau
        :x_tau: initial image
        :z_t: current noisy instance
        :x_t: current clean instance
        :alpha_t: time dependant constant
        :scale_grads: scale grads based on time dependant constant
        """

        z_in = nn.Parameter(z_t.detach().requires_grad_(True))
        x_in = nn.Parameter(x_t.detach().requires_grad_(True))

        z_img = self.latent_decoder(z_in)
        x_img = self.latent_decoder(x_in)
        try:
            m1 = (
                l1_loss * torch.norm(z_img - x_tau.to(z_in.device), p=1, dim=1).sum()
                if l1_loss != 0
                else 0
            )
            m2 = (
                l2_loss * torch.norm(z_img - x_tau.to(z_in.device), p=2, dim=1).sum()
                if l2_loss != 0
                else 0
            )
        except:
            breakpoint()
        mv = l_perc(x_in, x_tau) if l_perc is not None else 0

        if isinstance(m1 + m2 + mv, int):
            return 0

        if isinstance(m1 + m2, int):
            grads = 0
        else:
            grads = torch.autograd.grad(m1 + m2, z_in)[0]

        if isinstance(mv, int):
            return grads
        else:
            if scale_grads:
                return grads + torch.autograd.grad(mv, x_in)[0] / alpha_t
            else:
                return grads + torch.autograd.grad(mv, x_in)[0]

    @torch.no_grad()
    def generate_mask(self, x1, x2, dilation):
        """
        Extracts a mask by binarizing the difference between
        denoised image at time-step t and original input.
        We generate the mask similar to ACE.

        :x1: denoised image at time-step t
        :x2: original input image
        :dilation: dilation parameters
        """
        assert (dilation % 2) == 1, "dilation must be an odd number"
        x1 = (x1 + 1) / 2
        x2 = (x2 + 1) / 2
        mask = (x1 - x2).abs().sum(dim=1, keepdim=True)
        mask = mask / mask.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        dil_mask = F.max_pool2d(mask, dilation, stride=1, padding=(dilation - 1) // 2)
        return mask, dil_mask

    def FastDiME(
        self,
        img,
        inpaint: float,
        dilation: float,
        t: float,
        guided_iterations: int,
        class_grad_kwargs: dict,
        dist_grad_kargs: dict,
        explainer_config,
        scale_grads: bool = False,
        boolmask_in=None,
    ):
        boolmask = None
        to_pil = ToPILImage()
        class_grad_fn = (
            self.clean_class_cond_fn if "Multiclass" else self.clean_multiclass_cond_fn
        )

        class_grad_fn = self.clean_multiclass_cond_fn
        dist_fn = self.dist_cond_fn
        guidance_scale = self.guidance_scale
        num_inference_steps = self.steps_number
        do_classifier_free_guidance = self.classifier_free_guidance
        self_optimized_masking = self.self_optamized_masking
        # Prepare timesteps.
        sigmas = self.prepare_time_steps(
            num_inference_steps=num_inference_steps, end_time=t * 1000, shift=self.shift
        ).to(self.device)
        to_pil(img.squeeze(0)).save(
            "/home/a.zeid/peal_private/utils/delete_me/original_img.png"
        )
        # Encode image into latent space with the VAE
        height, width = self.config.data.input_size[1:]
        vae_scale_factor = self.pipe.vae_scale_factor
        with torch.no_grad():
            latents = self.latent_encoder(img.to(self.device))
        # Initialize x_t as the current latent (will be updated iteratively)
        x_t = latents.clone()
        noise = torch.randn_like(x_t, device=self.device)
        # self.latent_decoder(noise, img=True).save(
        #     "/home/a.zeid/peal_private/utils/delete_me/noise.png"
        # )
        z_t = t * noise + (1 - t) * x_t
        self.latent_decoder(z_t, img=True).save(
            "/home/a.zeid/peal_private/utils/delete_me/noised_image.png"
        )
        x_t_steps = []
        z_t_steps = []
        if do_classifier_free_guidance:
            x_t = torch.cat([x_t] * 2)
            z_t = torch.cat([z_t] * 2)

        # encoding prompts
        prompt = self.prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        # Main iterative denoising loop in latent space
        for i in range(num_inference_steps):
            x_t_steps.append(x_t.detach().cpu().clone())
            z_t_steps.append(z_t.detach().cpu().clone())
            t = self.rescale_time_step(sigmas[i])
            timestep = t.expand(x_t.shape[0])
            # Use the Transformer to predict the noise residual.
            self.pipe.transformer.enable_gradient_checkpointing()
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=z_t,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # noise_pred = -guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

            # Compute guidance gradients:
            grads = 0
            if class_grad_fn is not None:
                xt_in = torch.clone(x_t.detach())
                # visualzing the original image gradient
                # if i == 0:

                #     grads_org = class_grad_fn(
                #         x_t=img.to("cuda").requires_grad_(True),
                #         resize=True,
                #         **class_grad_kwargs,
                #     )
                #     ref = torch.zeros_like(grads_org[0])
                #     grad_img = high_contrast_heatmap(ref, -grads_org)
                # to_pil(grad_img[0][0]).save(
                #     f"/home/a.zeid/peal_private/utils/delete_me/grads_class_org_img.png"
                # )
                # del grads_org
                class_grad = class_grad_fn(
                    x_t=xt_in,
                    resize=False,
                    threshold=self.grad_threshold,
                    **class_grad_kwargs,
                ) / (t + 5.0e-3 if scale_grads else 1)
                grads += class_grad
                # visualize the gradient
                print("grads norm", grads.norm(p=float("inf")))
                # self.latent_decoder(grads[0])
                ref = torch.zeros_like(grads[0]).to("cpu")
                # grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))
                # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                #     f"/home/a.zeid/peal_private/utils/delete_me/grads_class{i}.png"
                # )

                if dist_fn is not None:
                    x_t_in = torch.clone(x_t.detach())
                    z_t_in = torch.clone(z_t.detach())
                    dist_grad = self.dist_cond_fn(
                        x_tau=img,
                        z_t=z_t_in,
                        x_t=x_t_in,
                        alpha_t=(1 - t + 5.0e-3),
                        scale_grads=False,
                        **dist_grad_kargs,
                    )
                    ref = torch.zeros_like(grads).to("cpu")
                    # grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))

                    # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/grads_class_dist{i}.png"
                    # )
                    grads = grads + dist_grad
                    # visualize the gradient
                    # print("grads1 norm", grads.norm(p=float("inf")))

                    # grads_pixels = self.latent_decoder(grads).squeeze(0)
                    ref = torch.zeros_like(grads).to("cpu")
                    # grad_img = high_contrast_heatmap(ref, -grads.to("cpu"))

                    # self.latent_decoder(grad_img[0].to("cuda"), img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/grads_class_dist_plus{i}.png"
                    # )
                    """to_pil(grad_img[0]).save(
                        f"/home/a.zeid/peal_private/utils/delete_me/grads_class_dist{i}.png"
                    )"""
            # scaling the gradeint
            """if scale_grads:
                grads = grads / (t.float().mean() + 1e-5)"""
            dt = sigmas[i + 1] - sigmas[i]
            z_t = z_t + dt * noise_pred
            x_t = z_t - (sigmas[i]) * noise_pred  # / (1 - sigmas[i])
            # visualizing the results
            # self.latent_decoder(z_t, img=True).save(
            #     f"/home/a.zeid/peal_private/utils/delete_me/z_t{i}.png"
            # )
            # self.latent_decoder(noise_pred, img=True).save(
            #     f"/home/a.zeid/peal_private/utils/delete_me/noise_pred{i}.png"
            # )
            # self.latent_decoder(x_t, img=True).save(
            #     f"/home/a.zeid/peal_private/utils/delete_me/x_t{i}.png"
            # )

            norm = grads.norm(p=float("inf"))
            # print("norm", norm)
            if norm > explainer_config.gradient_clipping:
                print("gradient clipping")
                rescale_factor = explainer_config.gradient_clipping / norm
                grads = grads * rescale_factor

            # different way to move the calcualte the grad in pixel sapce and move it to the VAE space, does not work
            """grad = torch.nn.functional.interpolate(
                grads[0],
                size=(height // vae_scale_factor, width // vae_scale_factor),
            )
            grads = self.latent_encoder(grads)"""

            """z_t = self.latent_encoder(z_t)
            breakpoint()
            self.latent_decoder(z_t, img=True).save(
                f"/home/a.zeid/peal_private/utils/delete_me/z_t_remove_grad{i}.png"
            )"""

            z_t = z_t - class_grad_kwargs["lr"] * grads
            # self.latent_decoder(z_t, img=True).save(
            #     f"/home/a.zeid/peal_private/utils/delete_me/z_t_after_grad{i}.png"
            # )
            # Apply FASTDiME self-optimized masking if enabled:
            if self_optimized_masking:
                print("self optimized mask")
                # extract time-depedent mask (Eq. 6)
                with torch.no_grad():
                    x_0_denoised = self.latent_decoder(x_t.detach())
                # mask_t, dil_mask = generate_smooth_mask(
                #     img.to(self.device), x_0_denoised, explainer_config.dilation
                # )
                mask_t, dil_mask = self.generate_mask(
                    img.to(self.device), x_0_denoised, explainer_config.dilation
                )

                boolmask = (dil_mask < explainer_config.inpaint).float()
                height, width = x_0_denoised.shape[2:]

                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )

                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = (
                        torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                    )
                    # self.latent_decoder(noise, img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/noise_boolmask{i}.png"
                    # )
                    # z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
                    # self.latent_decoder(z_t, img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/zt_after_boolmask{i}.png"
                    # )
            # apply masking with fixed mask when available with our 2-step approaches
            if boolmask_in is not None:
                # fixed mask

                height, width = img.shape[2:]
                boolmask = boolmask_in
                boolmask_latent = torch.nn.functional.interpolate(
                    boolmask,
                    size=(height // vae_scale_factor, width // vae_scale_factor),
                )
                # masking denoised and sampled images (Eq. 7 & 8)
                with torch.no_grad():
                    x_t = x_t * (1 - boolmask_latent) + boolmask_latent * latents
                    noise = (
                        torch.randn_like(z_t) * sigmas[i] + (1 - sigmas[i]) * latents
                    )
                    # self.latent_decoder(noise, img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/noise_boolmask{i}.png"
                    # )
                    # self.latent_decoder(boolmask_latent * noise, img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/noise_x_boolmask{i}.png"
                    # )
                    # self.latent_decoder(z_t * (1 - boolmask_latent), img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/zt_x_boolmask{i}.png"
                    # )
                    # z_t = z_t * (1 - boolmask_latent) + boolmask_latent * noise
                    # self.latent_decoder(z_t, img=True).save(
                    #     f"/home/a.zeid/peal_private/utils/delete_me/zt_after_boolmask{i}.png"
                    # )

                # allowing to set a threshold to stop after reaching oin target confidence
                """with torch.no_grad():
                    predictor = class_grad_kwargs["classifier"]
                    counterfactual = self.latent_decoder(z_t)
                    preds = torch.nn.Softmax(dim=-1)(
                        predictor(counterfactual.to(self.device)).detach().cpu()
                    )
                    print("pred", preds)
                    y_target_end_confidence = torch.zeros([img.shape[0]])
                    target_classes = class_grad_kwargs["y"]
                    for i in range(img.shape[0]):

                        y_target_end_confidence[i] = preds[i, target_classes[i]]
                        if y_target_end_confidence[i] > 0.5:"""
            # clean up some variables if needed
            """del (x_t_in, z_t_in, noise, boolmask, boolmask_latent, x_0_denoised)"""
        with torch.no_grad():
            if boolmask is not None:

                z_t = z_t * (1 - boolmask_latent) + boolmask_latent * latents
                self.latent_decoder(z_t, img=True).save(
                    f"/home/a.zeid/peal_private/utils/delete_me/zt_final{i}.png"
                )
            final_image = self.latent_decoder(z_t)

        return final_image, x_t_steps, z_t_steps


def create_label_image(text, image_size, font_size):
    """
    Create a tensor containing a text label image.

    Args:
        text (str): Text to display.
        image_size (tuple): Tuple (C, H, W) of the target image size.
        font_size (int): Size of the font.

    Returns:
        torch.Tensor: Tensor of shape (1, C, H, W).
    """
    transform = transforms.ToTensor()
    C, H, W = image_size
    img = Image.new("RGB" if C == 3 else "L", (W, H), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        # text_bbox = draw.textbbox((0, 0), text, font=font)
        # text_w = text_bbox[2] - text_bbox[0]
        # text_h = text_bbox[3] - text_bbox[1]
        # x = (W - text_w) / 2
        # y = (H - text_h) / 2
    except IOError:
        font = ImageFont.load_default(font_size)
    w = draw.textlength(text, font=font)
    h = font_size
    x = (W - w) / 2
    y = (H - h) / 2

    draw.text((x, y), text, fill="black" if C == 3 else 0, font=font)

    label_tensor = transform(img)
    return label_tensor.unsqueeze(0)


def load_yml_file(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise


def create_image_grid(images, transform, grid_size=None):
    """
    Create and display a grid of images with no empty spots.

    Args:
        images (list): A list of images to include in the grid.
        transform (callable): A transformation function to apply to each image.
        grid_size (tuple, optional): Number of rows and columns in the grid (rows, cols).

    Returns:
        None
    """
    transformed_images = [transform(img) for img in images]

    # Dynamically determine grid size if not provided
    num_images = len(transformed_images)
    if grid_size is None:
        rows = int(num_images**0.5)
        cols = (num_images + rows - 1) // rows
    else:
        rows, cols = grid_size

    # Create the grid
    grid = make_grid(transformed_images, nrow=cols, padding=2)

    # Convert the grid to a PIL image and display
    grid_image = to_pil_image(grid)
    plt.figure(figsize=(cols * 2, rows * 2))
    plt.imshow(grid_image)
    plt.axis("off")
    plt.show()


def visualize_results(images, fp, font_size):
    """
    Visualize and save a collage of images with labels.

    Args:
        images (dict): Dictionary of images with their names as keys.
        fp (str): File path to save the collage.
        font_size (int): Font size for labels.

    Returns:
        None
    """
    for image in images:
        if isinstance(images[image], Image.Image):
            transform = transforms.ToTensor()
            images[image] = transform(images[image]).unsqueeze(0)
        if (images[image], torch.Tensor):
            if images[image].dim() < 4:
                images[image] = images[image].unsqueeze(0)
    components = [(images[i].to("cpu"), i) for i in images]

    rows = []
    for tensor, name in components:
        B, C, H, W = tensor.shape
        label = create_label_image(name, (C, H, W), font_size=font_size)
        label = label.to(tensor.device)
        row = torch.cat([label, tensor], dim=0)  # Shape: (B+1, C, H, W)
        rows.append(row)

    save_tensor = torch.cat(rows, dim=0)
    torchvision.utils.save_image(
        save_tensor, fp=os.path.join(fp, "collage.png"), nrow=B + 1
    )
