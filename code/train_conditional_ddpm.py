from project_cli import resolve_training_config, training_parser
from ddpm_trainer import train_diffusion


def main() -> None:
    parser = training_parser(
        "Train a class-conditional MNIST DDPM with classifier-free guidance.",
        "configs/mnist_conditional.json",
    )
    config = resolve_training_config(parser, dataset="mnist", conditional=True)
    train_diffusion(config)


if __name__ == "__main__":
    main()

