"""Entry point for the store simulation."""

import argparse
from simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Clothing store simulation")
    parser.add_argument("--hours",        type=int,   default=8,               help="Hours to simulate (default: 8)")
    parser.add_argument("--arrival-prob", type=float, default=0.30,            help="Peak P(new shopper per minute) (default: 0.30)")
    parser.add_argument("--open-hour",    type=int,   default=10,              help="Store opening hour 0-23 (default: 10)")
    parser.add_argument("--seed",         type=int,   default=None,            help="Random seed for reproducibility")
    parser.add_argument("--output",       type=str,   default="store_log.csv", help="Output CSV filename")
    args = parser.parse_args()

    Simulation(
        duration_hours=args.hours,
        arrival_prob=args.arrival_prob,
        open_hour=args.open_hour,
        seed=args.seed,
        output_file=args.output,
    ).run()


if __name__ == "__main__":
    main()
