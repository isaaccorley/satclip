import argparse

from satclip.location_encoder_only import export_location_encoder_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Extract a compact SatCLIP location-only checkpoint."
    )
    parser.add_argument("source_checkpoint", help="Path to a full SatCLIP Lightning checkpoint.")
    parser.add_argument(
        "output_checkpoint",
        help="Path to write the compact location-only checkpoint.",
    )
    args = parser.parse_args()
    export_location_encoder_checkpoint(args.source_checkpoint, args.output_checkpoint)


if __name__ == "__main__":
    main()
