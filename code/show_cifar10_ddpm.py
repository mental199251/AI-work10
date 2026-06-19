from result_generation import display_parser, generate_standard_results


def main() -> None:
    parser = display_parser(
        "Generate optional CIFAR-10 DDPM visualizations.",
        "outputs/cifar10/latest.pt",
        "outputs/cifar10_results",
    )
    generate_standard_results("cifar10", parser.parse_args())


if __name__ == "__main__":
    main()

