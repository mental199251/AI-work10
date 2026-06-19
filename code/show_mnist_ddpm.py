from result_generation import display_parser, generate_standard_results


def main() -> None:
    parser = display_parser(
        "Generate required MNIST DDPM visualizations.",
        "outputs/mnist_baseline/latest.pt",
        "outputs/mnist_results",
    )
    generate_standard_results("mnist", parser.parse_args())


if __name__ == "__main__":
    main()

