from pathlib import Path
import cv2
import time

from .steps import (
    Binarization,
    NoiseRemoval,
)


class ImagePreprocessingPipeline:

    def __init__(self, logger):
        self.logger = logger


    def process(self, image, image_name, output_dir):

        configs = []


        for b in [
            "none",
            "basic",
            "otsu",
            "adaptive_mean",
            "adaptive_gaussian",
            "yannihorne",
            "niblack"
        ]:

            for n in [
                "none",
                "mean_filter",
                "median_filter",
                "gaussian_filter",
                "unsharp_filter",
                # "conservative_filter",
                "laplacian_filter",
                "frequency_filter",
                # "crimmins_speckle_removal"
            ]:

                configs.append((b,n))

        self.save_configurations(
            configs,
            output_dir
        )

        outputs=[]


        for idx,(bin_step,noise_step) in enumerate(configs):

            start=time.time()


            self.logger.info(
                f"[{image_name}] "
                f"Config {idx+1}/{len(configs)} | "
                f"Binarization={bin_step} | "
                f"NoiseRemoval={noise_step}"
            )


            try:

                img=image.copy()


                img=getattr(
                    Binarization,
                    bin_step
                )(img)


                img=getattr(
                    NoiseRemoval,
                    noise_step
                )(img)


                elapsed=time.time()-start


                self.logger.info(
                    f"[{image_name}] "
                    f"Config {idx} finished "
                    f"in {elapsed:.3f}s"
                ) # TODO: Save in parquet


                outputs.append(
                    {
                        "config": idx,
                        "binarization": bin_step,
                        "noise": noise_step,
                        "image": img,
                        "time": elapsed
                    }
                )


            except Exception as e:

                self.logger.exception(
                    f"[{image_name}] "
                    f"FAILED config {idx}: {e}"
                )


        return outputs



    def run(
        self,
        input_dir:Path,
        output_dir:Path,
        resume=True
    ):


        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        processed=0


        for file in input_dir.iterdir():


            if file.suffix.lower() not in [
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff"
            ]:
                continue


            self.logger.info(
                f"Starting preprocessing: {file.name}"
            )


            image=cv2.imread(
                str(file)
            )


            if image is None:

                self.logger.warning(
                    f"Could not read {file}"
                )

                continue



            results=self.process(
                image,
                file.name,
                output_dir
            )



            for result in results:


                out = output_dir / (
                    f"{file.stem}_config_{result['config']}.tiff"
                )


                cv2.imwrite(
                    str(out),
                    result["image"]
                )


                # self.logger.info(
                #     f"Saved {out.name}"
                # )



            processed+=1



        self.logger.info(
            f"Finished preprocessing. "
            f"Images processed: {processed}"
        )


        return processed
    
    def save_configurations(
        self,
        configs,
        output_dir: Path
    ):
        config_file = output_dir / "configurations.txt"

        with open(config_file, "w") as f:

            for idx, (bin_step, noise_step) in enumerate(configs):

                f.write(
                    f"config_{idx}\n"
                )

                f.write(
                    f"  binarization: {bin_step}\n"
                )

                f.write(
                    f"  noise_removal: {noise_step}\n"
                )

                f.write(
                    "\n"
                )


        self.logger.info(
            f"Saved preprocessing configurations to {config_file}"
        )