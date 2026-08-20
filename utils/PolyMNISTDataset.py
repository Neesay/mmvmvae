import numpy as np
import argparse
import torch
import os
import glob

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image

# from scipy import ndimage


class PolyMNIST(Dataset):
    """Multimodal MNIST Dataset."""

    def __init__(self, dir_data, num_views, mod_order, transform=None, target_transform=None,
                 cache=False):
        """
        Args:
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
            cache: preload every view into RAM as a uint8 tensor at construction
                time, so __getitem__ never touches the filesystem or the PNG
                decoder again. Costs ~12 kB/sample (num_views * 3 * 28 * 28),
                i.e. ~118 MB for the 10k-sample test split.
        """
        super().__init__()
        self.num_modalities = num_views
        self.dir_data = dir_data
        self.transform = transform
        self.target_transform = target_transform
        self.label_names = "digit"
        mod_list = [n for n in range(num_views)]
        mod_list.remove(mod_order)
        self.modalities_order = [mod_order] + mod_list

        # save all paths to individual files
        # sorted(): glob returns entries in arbitrary filesystem order, which is
        # not guaranteed to agree between the m0/, m1/, ... directories even
        # though they contain identically-named files. __getitem__ pairs the
        # views purely by list position and reports a single shared label, so an
        # order mismatch would silently pair a "3" in one view with a "7" in
        # another. The filenames are identical across modality dirs, so sorting
        # lines them up by construction (and makes sample order reproducible).
        self.file_paths = {dp: [] for dp in range(self.num_modalities)}
        for dp in range(self.num_modalities):
            files = sorted(
                glob.glob(os.path.join(self.dir_data, "m" + str(dp), "*.png"))
            )
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

        # Labels are encoded in the filename ("<index>.<digit>.png") and are
        # shared across views, so parse them once here rather than splitting 5
        # strings on every __getitem__ call (4 of which were then discarded).
        self.labels = [
            int(os.path.basename(fp).split(".")[-2]) for fp in self.file_paths[0]
        ]
        # cheap guard that the sorted pairing really does line the views up
        for dp in range(1, self.num_modalities):
            for i, fp in enumerate(self.file_paths[dp]):
                if int(os.path.basename(fp).split(".")[-2]) != self.labels[i]:
                    raise RuntimeError(
                        f"view m{dp} in {self.dir_data} disagrees with m0 on the label "
                        f"at index {i}; the views are misaligned"
                    )

        self.cache = None
        if cache:
            self._build_cache()

    def _build_cache(self):
        """Decode every PNG once into a (views, samples, 3, 28, 28) uint8 tensor."""
        cache = torch.empty(
            (self.num_modalities, self.num_files, 3, 28, 28), dtype=torch.uint8
        )
        for dp in range(self.num_modalities):
            for i, fp in enumerate(self.file_paths[dp]):
                with Image.open(fp) as img:
                    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)  # HWC
                cache[dp, i] = torch.from_numpy(arr).permute(2, 0, 1)
        self.cache = cache
        print(
            f"cached {self.num_files} samples x {self.num_modalities} views from "
            f"{self.dir_data} ({cache.numel() / 1e6:.0f} MB uint8)"
        )

    @staticmethod
    def _create_mmnist_dataset(
        savepath,
        backgroundimagepath,
        num_modalities,
        train,
        rotate_mnist=False,
        translate_mnist=False,
    ):
        """Created the Multimodal MNIST Dataset under 'savepath' given a directory of background images.

        Args:
            savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                exist.
            backgroundimagepath (str): path to a directory filled with background images. One background images is
                used per modality.
            num_modalities (int): number of modalities to create.
            train (bool): create the dataset based on MNIST training (True) or test data (False).
            rotate_mnist (bool): add a random rotation [-45, ..., 45] degree to each digit.
            translate_mnist (bool): downsample MNIST by a factor of 2 and place it at a random x/y-coordinate

        """

        # load MNIST data
        mnist = datasets.MNIST("/tmp", train=train, download=True, transform=None)

        # load background images
        background_filepaths = sorted(
            glob.glob(os.path.join(backgroundimagepath, "*.jpg"))
        )  # TODO: handle more filetypes
        print("\nbackground_filepaths:\n", background_filepaths, "\n")
        if num_modalities > len(background_filepaths):
            raise ValueError(
                "Number of background images must be larger or equal to number of modalities"
            )
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(num_modalities):
            unimodal_path = os.path.join(savepath, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(num_modalities):
                ixs_perm = ixs[
                    torch.randperm(len(ixs))
                ]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = PolyMNIST._add_background_image(
                        background_images[m],
                        mnist.data[ix],
                        rotate_mnist=rotate_mnist,
                        translate_mnist=translate_mnist,
                    )
                    # save as png
                    filepath = os.path.join(savepath, "m%d/%d.%d.png" % (m, i, digit))
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        print(
                            "Saved %d/%d images to %s"
                            % (cnt, len(mnist) * num_modalities, savepath)
                        )
        assert cnt == len(mnist) * num_modalities

    @staticmethod
    def _add_background_image(
        background_image_pil,
        mnist_image_tensor,
        change_colors=False,
        rotate_mnist=False,
        translate_mnist=False,
    ):
        # rotate mnist image
        if rotate_mnist is True:
            deg = np.random.randint(-45, 45)
            mnist_image_pil = Image.fromarray(
                mnist_image_tensor.squeeze(0).numpy(), mode="L"
            )
            mnist_image_pil_rotated = mnist_image_pil.rotate(deg)
            mnist_image_tensor_rotated = (
                transforms.ToTensor()(mnist_image_pil_rotated) * 255.0
            )
            mnist_image_tensor = mnist_image_tensor_rotated

        # translate mnist image: downsamle digit and place it at a random location
        if translate_mnist is True:
            mnist_image_tensor_downsampled = torch.nn.functional.interpolate(
                mnist_image_tensor.unsqueeze(0).float(),
                scale_factor=0.75,
                mode="bilinear",
            )
            mnist_image_tensor = mnist_image_tensor * 0  # black out everything
            x = np.random.randint(0, 21)
            y = np.random.randint(0, 21)
            mnist_image_tensor[:, x : x + 21, y : y + 21] = (
                mnist_image_tensor_downsampled
            )

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(
            torch.bool
        )  # NOTE: mnist is _not_ normalized to [0, 1]

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError(
                "Unexpected dimensionality of MNIST image:", img_binarized.shape
            )

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        if self.cache is not None:
            # The cache holds exactly what ToTensor() produces apart from the
            # final uint8 -> float scaling (kept as uint8 to cut cache memory
            # 4x), so dividing by 255 here reproduces ToTensor() exactly.
            # uint8 / float promotes to a *new* float32 tensor, so this can
            # never alias (and therefore never corrupt) the cache itself.
            images_dict = {
                "m%d" % m: self.cache[m, index] / 255.0
                for m in range(self.num_modalities)
            }
            return images_dict, self.labels[index]

        files = [self.file_paths[dp][index] for dp in range(self.num_modalities)]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]

        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]

        images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}
        # images_dict = {"m%d" % m: images[m] for m in self.modalities_order}

        return (
            images_dict,
            self.labels[index],
        )  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-modalities", type=int, default=5)
    parser.add_argument("--savepath-train", type=str, required=True)
    parser.add_argument("--savepath-test", type=str, required=True)
    parser.add_argument("--backgroundimagepath", type=str, required=True)
    parser.add_argument("--rotate-mnist", default=False, action="store_true")
    parser.add_argument("--translate-mnist", default=False, action="store_true")
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    PolyMNIST._create_mmnist_dataset(
        args.savepath_train,
        args.backgroundimagepath,
        args.num_modalities,
        train=True,
        rotate_mnist=args.rotate_mnist,
        translate_mnist=args.translate_mnist,
    )
    PolyMNIST._create_mmnist_dataset(
        args.savepath_test,
        args.backgroundimagepath,
        args.num_modalities,
        train=False,
        rotate_mnist=args.rotate_mnist,
        translate_mnist=args.translate_mnist,
    )
    print("Done.")
