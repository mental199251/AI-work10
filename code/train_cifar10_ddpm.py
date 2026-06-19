from project_cli import resolve_training_config, training_parser
from ddpm_trainer import train_diffusion


def main() -> None:
    parser = training_parser(
        "Train the optional CIFAR-10 DDPM.", "configs/cifar10.json"
    )
    config = resolve_training_config(parser, dataset="cifar10", conditional=False)
    train_diffusion(config)


if __name__ == "__main__":
    main()

